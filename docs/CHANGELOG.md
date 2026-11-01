# Change log and checkpoints

Every change pushed to GitHub is listed here. A checkpoint is a saved state of the
whole project with a name you can return to. Newest is at the top.

| Checkpoint | Date | What it added | On GitHub | Going back to the checkpoint below it removes |
|---|---|---|---|---|
| `checkpoint-3-website-phase-1` | 2026-11-01 | FloorCheck website: home page, plain wording, new look, new tabs, change log | Not yet | The new website. The old dashboard returns |
| `checkpoint-2-traffic-and-battery` | 2026-11-01 | Live floor traffic fixes, large-floor battery rules, corrected one-way and priority figures | Not yet | The traffic fixes and battery rules, and **brings back the withdrawn 5.8x claim** |
| `checkpoint-1-guide-sheet` | 2026-10-30 | State of the project after the final demo, with the guide's interaction sheet | Yes | Starting point for this log |

## Rules for every push

1. A new feature gets its own branch, `feature/<name>`, started from the latest
   checkpoint. It joins `multi-robot` and `main` only once the guide approves.
2. Each approved feature gets a row above and a `checkpoint-<number>-<name>` tag,
   pushed with the commit.
3. History is never rewritten: no force push, and existing tags are never moved.
4. Going back adds a new commit that undoes the work, so the old work stays
   recoverable and the record stays honest. The undo gets its own row here.

## Going back

**The guide rejects a feature that is still on its branch:** do nothing. It never
reached `main`, so the project and the live site are untouched.

**The guide rejects something already merged:** go back to the checkpoint before
it. This also removes everything added after that checkpoint, so check the table
first. To go back to `checkpoint-2-traffic-and-battery`, for example:

```bash
git revert --no-commit checkpoint-2-traffic-and-battery..HEAD
```

```bash
git restore --source=HEAD --staged --worktree docs/CHANGELOG.md
```

```bash
git commit -m "Go back to checkpoint-2-traffic-and-battery"
```

The middle command keeps this log, which would otherwise be undone too. Then add
a row for the undo.

**Removing one old feature while keeping newer ones** is only clean when they
touch different files. Otherwise the conflicts have to be resolved by hand, which
is why features stay on their own branches until they are approved.

To look at an old state without changing anything, use the command below, and
`git switch multi-robot` to come back.

```bash
git switch --detach checkpoint-2-traffic-and-battery
```
