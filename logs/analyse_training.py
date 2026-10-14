
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def main():
    run_dir = Path(input("Enter the training run directory: ").strip())

    if not run_dir.exists():
        print("Error: The specified path does not exist.")
        return

    if not run_dir.is_dir():
        print("Error: The specified path is not a directory.")
        return

    event_files = list(run_dir.glob("events.out.tfevents.*"))

    if not event_files:
        print("Error: No TensorBoard event files found.")
        return

    event_file = event_files[0]
    print(f"Event file found: {event_file}")

    zoom = input(
        "\nDo you want to zoom on an iteration range? (y/n): "
    ).strip().lower()

    if zoom == "y":
        try:
            start_step = int(input("Start iteration: "))
            end_step = int(input("End iteration: "))
        except ValueError:
            print("Error: Iterations must be integers.")
            return

        if start_step < 0 or end_step <= start_step:
            print("Error: Invalid iteration range.")
            return

        output_dir = (
            run_dir
            / "plots"
            / f"iterations_{start_step}_{end_step}"
        )

    elif zoom == "n":
        output_dir = run_dir / "plots"

    else:
        print("Error: Please enter 'y' or 'n'.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print("Loading TensorBoard events...", flush=True)

    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()

    print("Reading complete.", flush=True)

    scalar_tags = accumulator.Tags()["scalars"]

    print(f"\nFound {len(scalar_tags)} scalar metrics.")

    if zoom == "y":
        print(
            f"Generating plots for iterations "
            f"{start_step} to {end_step}..."
        )
    else:
        print("Generating plots for all iterations...")

    generated_count = 0

    for tag in scalar_tags:
        events = accumulator.Scalars(tag)

        if not events:
            continue

        if zoom == "y":
            events = [
                event
                for event in events
                if start_step <= event.step <= end_step
            ]

            if not events:
                print(f"Skipping {tag}: no data in selected range.")
                continue

        steps = [event.step for event in events]
        values = [event.value for event in events]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, values)

        plt.xlabel("Iteration")
        plt.ylabel(tag)
        plt.title(tag)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = tag.replace("/", "_").replace("\\", "_")
        output_path = output_dir / safe_name

        plt.savefig(output_path.with_suffix(".png"), dpi=200)

        plt.close()

        generated_count += 1
        print(f"Saved: {safe_name}.png")

    print(f"\nAnalysis complete.")
    print(f"Metrics plotted: {generated_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
