from rich.console import Console

console = Console()


def main() -> None:
    console.print("[bold green]Disaster Tweet Classifier CLI is working.[/bold green]")
    console.print("Available commands will be added later:")
    console.print("- download-data")
    console.print("- prepare-data")
    console.print("- train-baseline")
    console.print("- train")
    console.print("- evaluate")
    console.print("- export-onnx")
    console.print("- serve")
    console.print("- infer")
