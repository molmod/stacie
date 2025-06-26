#!/usr/bin/env bash
# Usage: .github/scripts/extract-notes.sh OWNER/SLUG GITREF

IFS='/'; read -ra REPOSITORY <<<"${1}"
OWNER=${REPOSITORY[0]}
SLUG=${REPOSITORY[1]}
GITREF=${2}

if [[ "${GITREF}" == refs/tags/* ]]; then
    TAG="${GITREF#refs/tags/}"
    VERSION="${TAG#v}"
    MACRO_MESO=$(echo "${VERSION}" | cut -d. -f1,2)
else
    TAG="unreleased"
    VERSION="Unreleased"
    MACRO_MESO="dev"
fi

# Extract the release notes from the changelog
sed -n "/## \[${VERSION}\]/, /## /{ /##/!p }" docs/source/development/changelog.md > notes.md

# Add a link to the release notes
URL="https://${OWNER}.github.io/${SLUG}/${MACRO_MESO}/changelog/#${TAG}"
echo "See [docs/changelog/#${TAG}](${URL}) for more details." >> notes.md

# Remove leading and trailing empty lines
sed -e :a -e '/./,$!d;/^\n*$/{$d;N;};/\n$/ba' -i notes.md
