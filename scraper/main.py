# IMPORTS
import os
import time
from datetime import datetime
from threading import Thread, Event, Lock
from typing import Optional

from rich import print, box
from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.text import Text, TextType

from labelling import label_sample
from virustotal import *

# CONSTANTS
CONSECUTIVE_FAILURE_COUNT = 3  # Don't continue processing if there are this many failures in a row

FAILURE_SLEEP_DURATION = 15
SUCCESS_SLEEP_DURATION = 5
NOT_YET_PROCESSED_SLEEP_DURATION = 3
SLEEP_INTERVAL = 0.05

# GLOBAL VARIABLES
# Thread variables
gStopEarly = Event()
gHashesLock = Lock()
gAllHashesLock = Lock()

gFinishCount = 0
gFinishCountLock = Lock()

# Display variables
gPanels = []
gPanelsLock = Lock()

# SET-UP
# Check that the folders exist
os.makedirs("../data/json", exist_ok=True)


# FUNCTIONS
def get_current_time() -> str:
    """
    Gets and formats the current time

    :return: current time formatted as a string
    """

    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_elapsed_time(start_time: datetime.time) -> str:
    """
    Gets the time from the `start_time` till now, and returns it as a formatted string.

    :param start_time: starting time
    :return: time elapsed
    """

    now = datetime.now()
    elapsed_time = now - start_time
    total_seconds = int(elapsed_time.total_seconds())

    return f"{total_seconds//3600:02d}:{total_seconds%3600 //60:02d}:{total_seconds%60:02d}"


def worker(worker_id: int, api_key: str):
    """
    Thread worker.

    :param worker_id: ID of the worker
    :param api_key: VirusTotal API key
    """

    global gFinishCount
    global all_hashes
    global hashes

    # Helper functions
    def update_panel(renderable: RenderableType, subtitle: Optional[TextType] = None):
        """
        Updates the panel with the specified renderable and subtitle.

        :param renderable: renderable to set on the panel
        :param subtitle: subtitle of the panel, defaults to None
        """

        gPanelsLock.acquire()
        gPanels[worker_id].renderable = renderable
        if subtitle is not None:
            gPanels[worker_id].subtitle = subtitle
        gPanelsLock.release()

    def pretty_sleep(
        duration: float,
        interval: float,
        group_renderable: RenderableType,
        prog_desc: str = "Timeout",
        panel_subtitle: Optional[TextType] = None,
    ):
        """
        Produces a nice sleep progress bar in the panel.

        :param duration: duration to sleep for
        :param interval: interval between updates
        :param group_renderable: renderable to display above the progress bar
        :param prog_desc: description for the progress bar, defaults to "Timeout"
        :param subtitle: subtitle of the panel, defaults to None
        """

        wait_progress = Progress()
        wait_task = wait_progress.add_task(prog_desc, total=duration // interval)

        update_panel(Group(group_renderable, wait_progress), subtitle=panel_subtitle)

        while not wait_progress.finished:
            time.sleep(interval)
            wait_progress.advance(wait_task)

    # Main code
    failure_count = 0
    success_count = 0
    while failure_count < CONSECUTIVE_FAILURE_COUNT and not gStopEarly.is_set():
        try:
            # Get the next hash to process
            gHashesLock.acquire()
            next_hash = hashes.pop()
            gHashesLock.release()

            # Try and get the report
            update_panel(
                Group(
                    Text.from_markup(f"[u b blue]Retrieving Report", justify="center"),
                    Text.from_markup(f"[blue]{next_hash}", justify="center"),
                )
            )

            success, report = get_report(next_hash, api_key)

            if not success:
                failure_count += 1
                if failure_count < CONSECUTIVE_FAILURE_COUNT:
                    pretty_sleep(
                        FAILURE_SLEEP_DURATION,
                        SLEEP_INTERVAL,
                        Text.from_markup(
                            f"[u b red]Failed ({failure_count}/{CONSECUTIVE_FAILURE_COUNT})", justify="center"
                        ),
                    )

                continue

            failure_count = 0

            if report is None:
                gAllHashesLock.acquire()
                all_hashes.remove(next_hash)
                gAllHashesLock.release()

                pretty_sleep(
                    NOT_YET_PROCESSED_SLEEP_DURATION,
                    SLEEP_INTERVAL,
                    Group(
                        Text.from_markup(f"[u b yellow]Notice", justify="center"),
                        Text.from_markup(f"[yellow]VirusTotal has not processed this yet", justify="center"),
                    ),
                )
                continue

            # Get the hashes from the report
            md5 = report["md5"]
            sha1 = report["sha1"]
            sha256 = report["sha256"]

            # Remove these hashes from the set
            gHashesLock.acquire()
            hashes = hashes.difference({md5, sha1, sha256})
            gHashesLock.release()

            # Try and guess the label of the sample
            label = label_sample(report)

            # Write the JSON report to disk
            with open(f"../data/json/{label}_{sha256}.json", "w") as f:
                json.dump(report, f, indent=4)

            # Report success
            success_count += 1
            pretty_sleep(
                SUCCESS_SLEEP_DURATION,
                SLEEP_INTERVAL,
                Text.from_markup(f"[b green]Success", justify="center"),
                panel_subtitle=f"Retrieved {success_count} reports"
            )

        except Exception as e:
            print(f"An error occurred in worker {worker_id}: {e}")
            update_panel(Text.from_markup(f"[r b red]An Error Occurred", justify="center"))
            break

    if failure_count >= CONSECUTIVE_FAILURE_COUNT:
        update_panel(Text.from_markup(f"[b red]Max Failures Hit", justify="center"))

    gFinishCountLock.acquire()
    gFinishCount += 1
    gFinishCountLock.release()


# MAIN
# Get API keys to use
api_keys = []
while True:
    key = input("Enter a VirusTotal API key here, or nothing to stop input: ").lower().strip()
    if len(key) == 0:
        break
    api_keys.append(key)

if len(api_keys) == 0:
    print("Please provide at least one API key.")
    exit(1)

print("\nUsing keys:")
for key in api_keys:
    print(f"- {key}")

# Get all the hashes that we have
with open("../data/hashes.txt", "r") as f:
    all_hashes = set([x.strip() for x in f.readlines()])

# Get all of the hashes that we have already processed
try:
    with open("../data/done-hashes.txt", "r") as f:
        done_hashes = set([x.strip() for x in f.readlines()])
except FileNotFoundError:
    done_hashes = set()

# Remove all the done hashes from the master list
hashes = all_hashes.difference(done_hashes)

threads = []
for i, api_key in enumerate(api_keys):
    # Create panels
    gPanels.append(Panel("Loading...", title=f"[b frame]Worker with key '{api_key[:8]}...'", subtitle=f"Retrieved 0 reports"))

    # Create the thread
    t = Thread(target=worker, args=(i, api_key))
    t.start()
    threads.append(t)

print("Press Ctrl + C to stop early.\n")

start_time = datetime.now()

process_columns = Columns(gPanels)
process_panel_group = Group(
    Align(Text(get_elapsed_time(start_time), style="green"), align="center"), Align(process_columns, align="center")
)
process_panel = Panel(process_panel_group, box=box.SIMPLE_HEAD, title=Text(get_current_time(), style="green"))

with Live(process_panel, refresh_per_second=10) as live:
    while not gStopEarly.is_set():
        gFinishCountLock.acquire()
        if gFinishCount >= len(api_keys):
            break
        gFinishCountLock.release()

        # Update times
        process_panel.title = Text(get_current_time(), style="green")
        process_panel_group.renderables[0] = Align(Text(get_elapsed_time(start_time), style="green"), align="center")

        try:
            time.sleep(0.25)
            live.update(process_panel)
        except KeyboardInterrupt:
            gStopEarly.set()
            break

print()

# If early stop, need to wait for threads to finish
if gStopEarly.is_set():
    with Progress() as thread_join_progress:
        thread_join_task = thread_join_progress.add_task(
            "Waiting for threads to join", total=(FAILURE_SLEEP_DURATION + SUCCESS_SLEEP_DURATION) // SLEEP_INTERVAL
        )
        while not thread_join_progress.finished:
            time.sleep(SLEEP_INTERVAL)
            thread_join_progress.advance(thread_join_task)

# Join threads
for t in threads:
    t.join()

# OUTPUT
print("Updating the completed hashes.")
update_master_list(all_hashes)
update_done_hashes()
print("All done.")
