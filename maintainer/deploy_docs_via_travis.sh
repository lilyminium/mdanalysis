#!/bin/bash

set -o errexit -o nounset

function die () {
    local msg="$1" err=${2:-1}
    echo "ERROR: $msg [$err]"
    exit $err
}

rev=$(git rev-parse --short HEAD)

# the following tests should be superfluous because of -o nounset
test -n "${GH_TOKEN}" || die "GH_TOKEN is empty: need OAuth GitHub token to continue" 100
test -n "${GH_REPOSITORY}" || die "GH_REPOSITORY must be set in .travis.yml" 100
test -n "${MDA_DOCDIR}" || die "MDA_DOCDIR must be set in .travis.yml" 100
test -n "${MAINTAIN_DIR}" || die "MAINTAIN_DIR must be set in .travis.yml" 100
test -n "${VERSION}" || die "VERSION must be set in .travis.yml" 100


cd ${MDA_DOCDIR} || die "Failed to 'cd ${MDA_DOCDIR}'. Run from the top level of the repository"

# move into $version subdirectory
mkdir ../${VERSION} && mv * ../${VERSION}

git init
git config user.name "${GIT_CI_USER}"
git config user.email "${GIT_CI_EMAIL}"

mv ../${VERSION} $VERSION

git remote add upstream git@github.com:lilyminium/mdanalysis.git
# git remote add upstream "https://${GH_TOKEN}@${GH_REPOSITORY}"
git fetch --depth 50 upstream ${GH_DOC_BRANCH} gh-pages
git reset upstream/gh-pages


touch .
touch .nojekyll

echo $VERSION
echo "ls *"
ls *
git add -A ${VERSION}/
git add .nojekyll

python ${MAINTAIN_DIR}/update_versions_json.py
git add versions.json

git status
# check for anything to commit
# https://stackoverflow.com/questions/3878624/how-do-i-programmatically-determine-if-there-are-uncommited-changes
# git diff-index --quiet HEAD -- || git commit -m "rebuilt html docs from branch ${GH_DOC_BRANCH} with sphinx at ${rev}"
git commit -m "rebuilt html docs from branch ${GH_DOC_BRANCH} with sphinx at ${rev}"
git push -q upstream HEAD:gh-pages

