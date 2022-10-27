# Git workflow

Notes on the workflow for version control used in this repository

__TLDR__ : Git Flow with GitHub

## Overview

[git-flow](https://nvie.com/posts/a-successful-git-branching-model/) is a popular development workflow for project using git for version control. The workflow outlined here is derived based on this work

The `main` branch will house working versions of the code with optional version tags with git-notes indicating details about the that particular version. For versioning [semantic versioning](https://semver.org/) format will be chosen

The `develop` branch will contain code currently in development

| Branch    | Description         |
| ------    | -----------         |
| main      | deployment ready    |
| develop   | feature development |
| hotfix    | bug fixes on master |

### Central Branches

As previously introduced `main` and `develop` are the primarly branches of the repo with `develop` also taking over the responsibilities usually taken over by `release`. An additional

As with most git workflows, the `main` branch represents the deployable code and `develop` contains

### Feature Branches

A new feature such as a new ANN application will be developed in a dedicated feature branch. This branch may be tracked on trello using the GitHub Power-Up

__TODO__ Come up with naming schemes for feature branches

### Taging

Tags and Notes maybe attached to commit merges into `main` from `develop` or `hotfix`. Naming tags should follow semantic versioning as mentioned previously

## Workflow

Each feature development such as adding an application or recipe should checkout from latest in `origin/develop`. This new branch will live as long as the feature development runs and then get merged back into `develop`

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

