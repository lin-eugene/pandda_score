import argparse
import gemmi
from pathlib import Path

def update_event_map_spacegroup(event_map_file: Path):
    ccp4 = gemmi.read_ccp4_map(str(event_map_file))

    ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    ccp4.setup()

    ccp4.update_ccp4_header(2, True)

    ccp4.write_ccp4_map(str(event_map_file))
    print(f"\tUpdated the event map at: {event_map_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update event map spacegroup')
    parser.add_argument('-e', '--event_map_path', type=str, required=True)

    args = parser.parse_args()
    update_event_map_spacegroup(Path(args.event_map_path))
