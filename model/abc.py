from console import console, error_console
import time
from rich.style import Style

style = "bold white on blue"


def do_work():
    for i in range(0, 10):
        print(i)
        time.sleep(1)


danger_style = Style(color="red", blink=True, bold=True)
console.print(":warning:", style=danger_style)
console.log("[red]Hello[/] world", style="encircle")
error_console.log("Error")
# console.print_json("[1,2,3,4]")
with console.status("Working...", spinner="monkey"):
    do_work()
console.rule("overflow")
console.log("Rich", style=style, justify="left")
console.log("Rich", style=style, justify="center")
console.log("Rich", style=style, justify="right")
from rich.progress import track

for i in track(range(20), description="Processing..."):
    time.sleep(1)  # Simulate work being done
# with open("report.txt", "wt") as report_file:
#     console = Console(file=report_file)
#     console.rule(f"Report Generated {datetime.now().ctime()}")
