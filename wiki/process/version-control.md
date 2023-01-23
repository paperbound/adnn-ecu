# Git workflow

Notes on the workflow for version control used in this repository

__TLDR__ : smaller Git Flow with GitHub

## Overview

[git-flow](https://nvie.com/posts/a-successful-git-branching-model/) is a popular development workflow for project using git for version control. The workflow outlined here is derived based on this work

The `main` branch will house working versions of the code with version tags and notes indicating details about the that particular version. For versioning [semantic versioning](https://semver.org/) format will be chosen

The `develop` branch will be used to coordinate feature development

| Branch    | Description                      |
| ------    | -----------                      |
| main      | deployment ready                 |
| develop   | coordinating feature development |

### Central Branches

As previously introduced `main` and `develop` are the primarly branches of the repo with `develop` also taking over the responsibilities usually taken over by `release`. As with most git workflows, the `main` branch represents the deployable code and `develop` contains features that may go into the next release into `master`

An additional `hotfix` branch is also present which is a longer lived version of the usual git-flow `hotfix-*` branches

### Feature Branches

A new feature such as a new ANN application will be developed in a dedicated feature branch. This branch may be tracked on trello using the GitHub Power-Up

### Taging

Notes and signed tags must be attached to commit merges into `main` from `develop` or `hotfix`. Naming tags should follow semantic versioning as previously mentioned

## Workflow

Each feature development such as adding an application or recipe should checkout from latest in `origin/develop`. This new branch will live as long as the feature development runs and then get merged back into `develop`. The merge should be made without fast-forwarding

Once a set of features are complete and vetted, a new version may be created by merging into `main` from `develop`

Hotfix checkouts from `main` should be checked out into `hotfix` and later be merged into `main`

### Pull Requests

The vetting process of feature branches will have Pull Request made on GitHub. This should be reviewed before merging back into `origin/develop`

__TODO__ Setup hooks for Pull Request, See Github Actions below

### Issue tracking

Issues will be tracked using the Github Issue Tracker

__TODO__ Write a template for issue write ups

### Github Actions

__TODO__ Setup some sensible actions as part of the workflow - such as make file scripts for NN application feature branches

## Submodules

Yocto layers will be added to the repo as submodules

Currently during project environment setup the `scripts/run-project-setup.sh` initialises and fetches the submodules

## Helpful Reading

- [Git best practises](https://sethrobertson.github.io/GitBestPractices/) | Seth Robertson

### Writing commit messages

- [Commit message formats](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) | Tim Pope
- [How to Write a Git Commit Message](https://cbea.ms/git-commit/) | Chris Beams
- [How to Write the perfect Pull Request](https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/) | Keavy McMinn