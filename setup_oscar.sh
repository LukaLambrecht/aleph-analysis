#!/usr/bin/bash

# Environment setup for Oscar, using an LCG view from /cvmfs/sft.cern.ch.
#
# Run as follows (on a compute node; /cvmfs is not mounted on login/vscode nodes):
#   source setup_oscar.sh            # latest LCG release with an el9 build
#   source setup_oscar.sh LCG_107    # specific LCG release
#
# Note: the Key4hep stack (/cvmfs/sw.hsf.org) currently fails to mount on Oscar,
# so this uses an LCG view instead, which provides ROOT, numpy, awkward, uproot,
# matplotlib, pandas, scipy, scikit-learn, onnx and onnxruntime.

_lcg_views=/cvmfs/sft.cern.ch/lcg/views

if [ -n "$KEY4HEP_STACK" ] || [ -n "$LCG_VERSION" ]; then
    echo "A Key4hep or LCG stack is already set up in this shell (${LCG_VERSION:-key4hep}),"
    echo "please start a new shell to avoid conflicts."
    unset _lcg_views
    return 1
fi

if [ ! -d "$_lcg_views" ]; then
    echo "ERROR: $_lcg_views is not accessible."
    echo "Note that /cvmfs is only available on Oscar compute nodes, not on login/vscode nodes."
    unset _lcg_views
    return 1
fi

# Find the setup script for a given release (el9, x86_64, gcc, optimized build;
# the newest gcc version is preferred).
_lcg_find_setup() {
    /usr/bin/ls -d "$_lcg_views/$1"/x86_64-el9-gcc*-opt 2>/dev/null | sort -V | tail -n1 |
        while read -r d; do [ -f "$d/setup.sh" ] && echo "$d/setup.sh"; done
}

_lcg_setup=""
if [ -n "$1" ]; then
    _lcg_setup=$(_lcg_find_setup "$1")
    if [ -z "$_lcg_setup" ]; then
        echo "ERROR: no x86_64-el9-gcc*-opt build found for LCG release '$1' in $_lcg_views."
    fi
else
    # Plain numbered releases only (e.g. LCG_107, LCG_106b), newest first.
    for _lcg_rel in $(/usr/bin/ls "$_lcg_views" | grep -E '^LCG_[0-9]+[a-z]?$' | sort -Vr); do
        _lcg_setup=$(_lcg_find_setup "$_lcg_rel")
        [ -n "$_lcg_setup" ] && break
    done
    if [ -z "$_lcg_setup" ]; then
        echo "ERROR: could not find any LCG release with an x86_64-el9-gcc*-opt build in $_lcg_views."
    fi
fi

if [ -n "$_lcg_setup" ]; then
    echo "Setting up LCG view: $(dirname "$_lcg_setup")"
    echo "To reproduce this environment, use: source setup_oscar.sh $(basename "$(dirname "$(dirname "$_lcg_setup")")")"
    source "$_lcg_setup"
    _lcg_status=0
else
    _lcg_status=1
fi

unset -f _lcg_find_setup
unset _lcg_views _lcg_setup _lcg_rel
eval "unset _lcg_status; return $_lcg_status"
