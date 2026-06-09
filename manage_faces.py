#!/usr/bin/env python3
"""
manage_faces.py — Inspect and edit the enrolled-faces database (faces.db).

    python manage_faces.py list                 # show everyone + sample counts
    python manage_faces.py remove "Eric"        # delete all of Eric's samples
    python manage_faces.py remove "Eric" -y     # skip the confirmation prompt

Enroll new people with enroll_faces.py. After any change, restart
stream_faces.py so it reloads the database.
"""

import argparse
import sys

import face_db


def cmd_list(_args) -> int:
    people = face_db.counts()
    if not people:
        print("No faces enrolled yet. Run: python enroll_faces.py \"Your Name\"")
        return 0
    print("Enrolled people:")
    for name, n in people.items():
        print(f"  {name:20s} {n} sample(s)")
    print(f"\n{len(people)} person(s), {sum(people.values())} sample(s) total.")
    return 0


def cmd_remove(args) -> int:
    people = face_db.counts()
    if args.name not in people:
        print(f"'{args.name}' is not enrolled. Names are case-sensitive — "
              f"run `manage_faces.py list` to see exact names.")
        return 1

    n = people[args.name]
    if not args.yes:
        reply = input(f"Delete all {n} sample(s) for '{args.name}'? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return 1

    deleted = face_db.delete_person(args.name)
    print(f"Removed '{args.name}' ({deleted} sample(s) deleted).")
    print("Restart stream_faces.py to apply the change.")
    return 0


def main() -> int:
    face_db.init_db()

    parser = argparse.ArgumentParser(
        description="Inspect and edit the enrolled-faces database."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List enrolled people and sample counts")

    p_remove = sub.add_parser("remove", help="Remove a person from the database")
    p_remove.add_argument("name", help="Exact enrolled name, e.g. \"Eric\"")
    p_remove.add_argument("-y", "--yes", action="store_true",
                          help="Skip the confirmation prompt")

    args = parser.parse_args()
    handlers = {"list": cmd_list, "remove": cmd_remove}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
