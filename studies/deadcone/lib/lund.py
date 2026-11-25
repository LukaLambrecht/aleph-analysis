############################################################
# Cambridge/Aachen reclustering and iterative declustering #
############################################################

# The constituents of each jet are reclustered with the e+e- Cambridge/Aachen algorithm
# (pairwise merging in order of increasing opening angle, E-scheme recombination).
# The resulting clustering tree is then declustered starting from the full jet,
# at each step recording the splitting and following one of the two branches:
#   - the branch containing a "tagged" constituent (e.g. a heavy-hadron proxy), if any;
#   - else the more energetic branch.
# Per splitting, the following quantities are recorded:
#   - theta: opening angle between the two branches,
#   - erad: energy of the radiator (i.e. of the parent node),
#   - z: energy fraction of the softer branch,
#   - kt: momentum of the softer branch times sin(theta),
#   - zemit: energy fraction of the branch that is not followed (the emission).
#
# The clustering is vectorized over jets (jets are padded to a common multiplicity),
# so it runs efficiently on large numbers of jets with numpy only.

import numpy as np
import awkward as ak


SPLITTING_FIELDS = ['theta', 'erad', 'z', 'kt', 'zemit']


def _cos_angle(p, q):
    # cosine of the angle between 3-vectors p (..., 3) and q (..., 3)
    num = np.sum(p*q, axis=-1)
    den = np.sqrt(np.sum(p*p, axis=-1) * np.sum(q*q, axis=-1))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.clip(num/den, -1, 1)


def decluster_padded(p4, valid, tagged):
    '''
    Recluster and decluster a set of jets given as padded numpy arrays.
    Input arguments:
    - p4: array of shape (njets, n, 4) with (px, py, pz, e) of the constituents.
    - valid: boolean array of shape (njets, n) indicating the actual (non-padding) constituents.
    - tagged: boolean array of shape (njets, n) indicating tagged constituents
      (the declustering follows the branch containing them).
    Returns:
    - dict of arrays of shape (njets, n-1) for each field in SPLITTING_FIELDS,
      in order of declustering (largest structures first),
    - boolean array of shape (njets, n-1) indicating valid splittings.
    '''
    njets, n, _ = p4.shape
    nsplit = max(n-1, 0)
    out = {f: np.zeros((njets, nsplit)) for f in SPLITTING_FIELDS}
    outvalid = np.zeros((njets, nsplit), dtype=bool)
    if n < 2 or njets == 0: return out, outvalid
    rows = np.arange(njets)

    # per-node information (nodes 0..n-1 are constituents, n+k is the k-th merging)
    nnodes = 2*n - 1
    node_p4 = np.zeros((njets, nnodes, 4))
    node_p4[:, :n] = p4
    node_tag = np.zeros((njets, nnodes), dtype=bool)
    node_tag[:, :n] = tagged & valid
    children = np.full((njets, n-1, 2), -1, dtype=int)
    angle = np.zeros((njets, n-1))

    # pairwise distances (1 - cos theta), with inf for invalid pairs and the diagonal
    mom = p4[:, :, :3].copy()
    dist = 1 - _cos_angle(mom[:, :, None, :], mom[:, None, :, :])
    pairvalid = valid[:, :, None] & valid[:, None, :]
    pairvalid[:, np.arange(n), np.arange(n)] = False
    dist[~pairvalid] = np.inf
    slot_node = np.tile(np.arange(n), (njets, 1))

    # clustering: merge the closest pair at each step
    for step in range(n-1):
        flat = np.argmin(dist.reshape(njets, -1), axis=1)
        dmin = dist.reshape(njets, -1)[rows, flat]
        active = np.isfinite(dmin)
        if not np.any(active): break
        a = rows[active]
        i, j = np.divmod(flat[active], n)
        newnode = n + step
        children[a, step, 0] = slot_node[a, i]
        children[a, step, 1] = slot_node[a, j]
        angle[a, step] = np.arccos(np.clip(1 - dmin[active], -1, 1))
        node_p4[a, newnode] = node_p4[a, slot_node[a, i]] + node_p4[a, slot_node[a, j]]
        node_tag[a, newnode] = node_tag[a, slot_node[a, i]] | node_tag[a, slot_node[a, j]]
        # slot i now holds the merged object, slot j is removed
        slot_node[a, i] = newnode
        mom[a, i] = node_p4[a, newnode, :3]
        dist[a, j, :] = np.inf
        dist[a, :, j] = np.inf
        alive = np.isfinite(dist[a, i, :]) | np.isfinite(dist[a, :, i])
        newdist = 1 - _cos_angle(mom[a, i][:, None, :], mom[a])
        dist[a, i, :] = np.where(alive, newdist, np.inf)
        dist[a, :, i] = dist[a, i, :]

    # declustering: start from the root (last merging) and follow the selected branch
    nconst = np.sum(valid, axis=1)
    node = np.where(nconst >= 2, n + nconst - 2, -1)
    for level in range(n-1):
        active = node >= n
        if not np.any(active): break
        a = rows[active]
        m = node[active] - n
        c1, c2 = children[a, m, 0], children[a, m, 1]
        e1, e2 = node_p4[a, c1, 3], node_p4[a, c2, 3]
        p1 = np.linalg.norm(node_p4[a, c1, :3], axis=-1)
        p2 = np.linalg.norm(node_p4[a, c2, :3], axis=-1)
        theta = angle[a, m]
        soft_is_1 = e1 < e2
        out['theta'][a, level] = theta
        out['erad'][a, level] = e1 + e2
        out['z'][a, level] = np.minimum(e1, e2) / (e1 + e2)
        out['kt'][a, level] = np.where(soft_is_1, p1, p2) * np.sin(theta)
        outvalid[a, level] = True
        # follow the tagged branch if exactly one is tagged, else the harder one
        t1, t2 = node_tag[a, c1], node_tag[a, c2]
        follow_1 = np.where(t1 != t2, t1, e1 >= e2)
        out['zemit'][a, level] = np.where(follow_1, e2, e1) / (e1 + e2)
        node[a] = np.where(follow_1, c1, c2)
    return out, outvalid


def _to_padded(jets_p4, jets_tagged, n):
    # convert jagged per-jet arrays (njets x var) to padded numpy arrays of length n
    def pad(x, fill):
        return ak.to_numpy(ak.fill_none(ak.pad_none(x, n, clip=True), fill))
    p4 = np.stack([pad(c, 0.) for c in jets_p4], axis=-1).astype(float)
    valid = pad(ak.ones_like(jets_p4[0], dtype=bool), False).astype(bool)
    tagged = pad(jets_tagged, False).astype(bool)
    return p4, valid, tagged


def decluster_jets(px, py, pz, e, tagged, max_pairs=2e7):
    '''
    Recluster and decluster jets given as jagged awkward arrays.
    Input arguments:
    - px, py, pz, e: awkward arrays of shape (event, jet, constituent).
    - tagged: boolean awkward array of the same shape (see decluster_padded).
    - max_pairs: maximum number of constituent pairs (njets * n^2) processed at once,
      controls the memory usage.
    Returns:
    - dict of awkward arrays of shape (event, jet, splitting) for each field in SPLITTING_FIELDS.
    '''
    njets_per_event = ak.num(px, axis=1)
    flat = [ak.flatten(x, axis=1) for x in (px, py, pz, e, tagged)]
    nconst = ak.to_numpy(ak.num(flat[0], axis=1))
    njets = len(nconst)
    nsplits = np.zeros(njets, dtype=int)

    # process jets in chunks of similar multiplicity to limit padding
    order = np.argsort(nconst, kind='stable')
    chunks = []
    start = 0
    while start < njets:
        n = max(int(nconst[order[start]]), 1)
        stop = start + 1
        while stop < njets:
            nmax = max(int(nconst[order[stop]]), 1)
            if (stop - start + 1) * nmax**2 > max_pairs: break
            stop += 1
        chunks.append(order[start:stop])
        start = stop

    # decluster each chunk and collect results per jet
    per_jet = {f: np.empty(njets, dtype=object) for f in SPLITTING_FIELDS}
    for idx in chunks:
        n = int(nconst[idx].max())
        jets = [x[idx] for x in flat]
        p4, valid, tag = _to_padded(jets[:4], jets[4], n)
        out, outvalid = decluster_padded(p4, valid, tag)
        counts = outvalid.sum(axis=1)
        nsplits[idx] = counts
        for f in SPLITTING_FIELDS:
            vals = out[f][outvalid] if out[f].size > 0 else np.zeros(0)
            parts = np.split(vals, np.cumsum(counts)[:-1]) if len(counts) > 0 else []
            for k, jetidx in enumerate(idx): per_jet[f][jetidx] = parts[k]

    # convert back to awkward arrays with structure (event, jet, splitting)
    result = {}
    for f in SPLITTING_FIELDS:
        vals = np.concatenate([v for v in per_jet[f]]) if njets > 0 else np.zeros(0)
        arr = ak.unflatten(vals, nsplits)
        result[f] = ak.unflatten(arr, njets_per_event)
    return result
