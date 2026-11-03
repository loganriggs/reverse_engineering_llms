"""Publish the committed, generated phone viewer to the small Pages branch.

Build the snapshot, commit docs/index.html on main, then run this script.
The branch contains only index.html and .nojekyll; this avoids Pages
checking out the entire large research repository for each viewer update.
"""

from __future__ import annotations

import subprocess


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def main():
    subprocess.run(['git', 'fetch', 'origin', 'gh-pages'], check=True)
    tree = git('rev-parse', 'HEAD:docs')
    names = git('ls-tree', '--name-only', tree).splitlines()
    assert sorted(names) == ['.nojekyll', 'index.html'], names
    parent = git('rev-parse', 'refs/remotes/origin/gh-pages')
    if tree == git('rev-parse', f'{parent}^{{tree}}'):
        print('Phone viewer already published')
        return
    commit = git('commit-tree', tree, '-p', parent, '-m',
                 'Update phone DGP viewer snapshot')
    subprocess.run(['git', 'push', 'origin',
                    f'{commit}:refs/heads/gh-pages'], check=True)
    print('Published phone snapshot', commit[:8])


if __name__ == '__main__':
    main()
