"""
Reusable batch pool manager for OpenAI Batch API.

Manages a rolling pool of concurrent batch jobs: maintains K batches in flight,
and when one completes, submits the next from the queue.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class BatchJob:
    """Represents a batch job in the pool."""
    file_path: str
    metadata: dict = field(default_factory=dict)
    batch_id: Optional[str] = None
    file_id: Optional[str] = None
    submitted_at: Optional[float] = None


class BatchPool:
    """
    Manages a rolling pool of OpenAI batch jobs.

    Usage:
        pool = BatchPool(
            openai_client=client,
            max_concurrent=5,
            poll_interval=30,
            on_complete=lambda job, results: save_results(...),
            on_fail=lambda job, error: handle_failure(...),
        )
        pool.add_job(batch_file_path, metadata={'run_idx': 0})
        pool.add_job(batch_file_path, metadata={'run_idx': 1})
        ...
        pool.run()  # blocks until all done
    """

    def __init__(
        self,
        openai_client,
        max_concurrent: int = 5,
        poll_interval: int = 30,
        on_complete: Callable[[BatchJob, list], None] = None,
        on_fail: Callable[[BatchJob, str], None] = None,
        description: str = "batch_pool",
    ):
        self.client = openai_client
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self.on_complete = on_complete or (lambda job, results: None)
        self.on_fail = on_fail or (lambda job, error: None)
        self.description = description

        # State
        self.pending: list[BatchJob] = []
        self.in_flight: dict[str, BatchJob] = {}  # batch_id -> BatchJob
        self.completed: list[str] = []  # batch_ids
        self.failed: list[str] = []  # batch_ids

        # Cache batch status to avoid repeated API calls in same poll cycle
        self._batch_cache: dict[str, object] = {}

    def add_job(self, batch_file_path: str, metadata: dict = None) -> None:
        """Add a job to the pending queue."""
        job = BatchJob(file_path=batch_file_path, metadata=metadata or {})
        self.pending.append(job)

    def _submit_job(self, job: BatchJob) -> str:
        """Upload batch file and create batch job. Returns batch_id."""
        with open(job.file_path, 'rb') as f:
            file_obj = self.client.files.create(file=f, purpose='batch')
        job.file_id = file_obj.id

        # OpenAI metadata values must be strings
        str_metadata = {k: str(v) for k, v in job.metadata.items()}
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": self.description, **str_metadata}
        )

        job.batch_id = batch.id
        job.submitted_at = time.time()
        return batch.id

    def _submit_next(self) -> Optional[str]:
        """Submit one job from pending queue. Returns batch_id or None."""
        if not self.pending:
            return None
        if len(self.in_flight) >= self.max_concurrent:
            return None

        job = self.pending.pop(0)
        batch_id = self._submit_job(job)
        self.in_flight[batch_id] = job
        return batch_id

    def _fill_pool(self) -> None:
        """Submit jobs until pool is full or pending is empty."""
        while len(self.in_flight) < self.max_concurrent and self.pending:
            self._submit_next()

    def _get_batch_status(self, batch_id: str) -> object:
        """Get batch status, using cache if available."""
        if batch_id not in self._batch_cache:
            self._batch_cache[batch_id] = self.client.batches.retrieve(batch_id)
        return self._batch_cache[batch_id]

    def _download_results(self, batch_id: str) -> list:
        """Download and parse batch results. Returns list of (custom_id, response) tuples."""
        batch = self._get_batch_status(batch_id)

        if not batch.output_file_id:
            raise RuntimeError(f"Batch {batch_id} has no output file")

        content = self.client.files.content(batch.output_file_id)

        import json
        results = []
        for line in content.text.strip().split('\n'):
            if line:
                obj = json.loads(line)
                results.append((obj['custom_id'], obj['response']))

        return results

    def _poll_all(self) -> tuple[list[str], list[str]]:
        """
        Poll all in-flight batches once.
        Returns (newly_completed_batch_ids, newly_failed_batch_ids).
        """
        # Clear cache for fresh status
        self._batch_cache.clear()

        newly_completed = []
        newly_failed = []

        for batch_id in list(self.in_flight.keys()):
            batch = self._get_batch_status(batch_id)
            status = batch.status

            if status == 'completed':
                newly_completed.append(batch_id)
            elif status in ('failed', 'expired', 'cancelled'):
                newly_failed.append(batch_id)

        return newly_completed, newly_failed

    def _handle_completed(self, batch_id: str) -> None:
        """Handle a completed batch: download results, call callback, cleanup."""
        job = self.in_flight.pop(batch_id)

        try:
            results = self._download_results(batch_id)
            self.on_complete(job, results)
            self.completed.append(batch_id)
        except Exception as e:
            # Download/callback failed - treat as failure
            self.on_fail(job, f"Failed to process results: {e}")
            self.failed.append(batch_id)

    def _handle_failed(self, batch_id: str) -> None:
        """Handle a failed batch: call callback, cleanup."""
        job = self.in_flight.pop(batch_id)
        batch = self._get_batch_status(batch_id)

        error_msg = f"Batch {batch.status}"
        if batch.errors and batch.errors.data:
            error_msg += f": {batch.errors.data[0].message}"

        self.on_fail(job, error_msg)
        self.failed.append(batch_id)

    def _print_status(self) -> None:
        """Print single-line status update."""
        slots = []
        for batch_id in self.in_flight:
            batch = self._get_batch_status(batch_id)
            completed = batch.request_counts.completed
            total = batch.request_counts.total
            slots.append(f"{completed}/{total}")

        # Pad to max_concurrent slots
        while len(slots) < self.max_concurrent:
            slots.append("----")

        done = len(self.completed)
        total_jobs = done + len(self.failed) + len(self.in_flight) + len(self.pending)
        queued = len(self.pending)
        failed_str = f" ({len(self.failed)} failed)" if self.failed else ""

        line = f"[{' '.join(slots)}] | Done: {done}/{total_jobs}{failed_str} | Queue: {queued}"
        print(f"\r{line}    ", end="", flush=True)  # extra spaces to clear previous longer lines

    def run(self) -> dict:
        """
        Run the batch pool until all jobs are complete.
        Returns summary dict with completed/failed counts.
        """
        total_jobs = len(self.pending)
        print(f"Starting batch pool: {total_jobs} jobs, max {self.max_concurrent} concurrent")

        # Initial submission
        self._fill_pool()
        self._print_status()

        # Main loop
        while self.in_flight:
            time.sleep(self.poll_interval)

            newly_completed, newly_failed = self._poll_all()

            # Handle completed batches
            for batch_id in newly_completed:
                self._handle_completed(batch_id)

            # Handle failed batches
            for batch_id in newly_failed:
                self._handle_failed(batch_id)

            # Submit more if we have capacity
            self._fill_pool()

            # Update status
            self._print_status()

        # Final newline after status line
        print()

        # Summary
        summary = {
            'total': total_jobs,
            'completed': len(self.completed),
            'failed': len(self.failed),
        }

        print(f"Batch pool complete: {summary['completed']}/{summary['total']} succeeded, {summary['failed']} failed")

        return summary
