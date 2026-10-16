import signal
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import UPLOAD_EVERY, DATA_DIR
from topics import sample_prompt
from generator import generate_responses
from uploader import save_example, upload_to_hub, load_stats, save_stats, load_seen_prompts

console = Console()
running = True
stats = {}


def handle_sigint(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, handle_sigint)


def build_display(stats: dict, current_topic: str, status: str, session_accepted: int) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")

    elapsed_secs = time.time() - stats.get("_start_ts", time.time())
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_secs))

    accepted = stats.get("total_accepted", 0)
    rate = session_accepted / (elapsed_secs / 3600) if elapsed_secs > 0 else 0

    table.add_row("Generated", str(stats.get("total_generated", 0)))
    table.add_row("Accepted", str(accepted))
    table.add_row("Skipped", str(stats.get("total_skipped", 0)))
    table.add_row("Uploads", str(stats.get("total_uploads", 0)))
    table.add_row("Rate", f"{rate:.1f} examples/hr")
    table.add_row("Elapsed", elapsed_str)
    table.add_row("Last upload", stats.get("last_upload") or "—")
    table.add_row("Topic", current_topic or "—")
    table.add_row("Status", Text(status, style="bold yellow"))

    return Panel(table, title="[bold green]Project Silkworm[/bold green]", border_style="green")


def main():
    global stats, running

    DATA_DIR.mkdir(exist_ok=True)
    stats = load_stats()
    stats["_start_ts"] = time.time()
    seen_prompts = load_seen_prompts()

    since_last_upload = 0
    session_accepted = 0
    current_topic = ""
    status = f"Starting... ({len(seen_prompts)} prompts already seen)"

    with Live(build_display(stats, current_topic, status, session_accepted), refresh_per_second=2, console=console) as live:
        while running:
            try:
                status = "Sampling prompt..."
                live.update(build_display(stats, current_topic, status, session_accepted))

                topic, prompt = sample_prompt()
                current_topic = topic

                if prompt in seen_prompts:
                    stats["total_skipped"] = stats.get("total_skipped", 0) + 1
                    status = f"Skipped duplicate: {topic}"
                    live.update(build_display(stats, current_topic, status, session_accepted))
                    continue

                status = f"Generating responses for: {topic}..."
                live.update(build_display(stats, current_topic, status, session_accepted))

                result = generate_responses(prompt)
                stats["total_generated"] = stats.get("total_generated", 0) + 1

                if result is None:
                    stats["total_skipped"] = stats.get("total_skipped", 0) + 1
                    status = "Skipped (responses too short or identical)"
                    save_stats(stats)
                    live.update(build_display(stats, current_topic, status, session_accepted))
                    continue

                chosen, rejected = result

                example = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "topic": topic,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                save_example(example)

                seen_prompts.add(prompt)
                stats["total_accepted"] = stats.get("total_accepted", 0) + 1
                session_accepted += 1
                since_last_upload += 1
                save_stats(stats)

                if since_last_upload >= UPLOAD_EVERY:
                    status = "Uploading to HuggingFace..."
                    live.update(build_display(stats, current_topic, status, session_accepted))
                    upload_to_hub(stats)
                    since_last_upload = 0

                status = f"Saved pair for: {topic}"
                live.update(build_display(stats, current_topic, status, session_accepted))

            except KeyboardInterrupt:
                running = False
            except Exception as e:
                status = f"Error: {e}"
                save_stats(stats)
                live.update(build_display(stats, current_topic, status, session_accepted))
                time.sleep(5)

        status = "Shutting down — uploading final batch..."
        live.update(build_display(stats, current_topic, status, session_accepted))

    if since_last_upload > 0:
        console.print("[bold yellow]Uploading final batch to HuggingFace...[/bold yellow]")
        upload_to_hub(stats)

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"Accepted: {stats.get('total_accepted', 0)} | "
        f"Generated: {stats.get('total_generated', 0)} | "
        f"Uploads: {stats.get('total_uploads', 0)}"
    )


if __name__ == "__main__":
    main()
