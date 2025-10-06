import pandas as pd
import numpy as np
from functions import *


def namematch(ref: pd.DataFrame,
              elecs: pd.DataFrame,
              name_col_ref="name",
              first_col_ref="first_name",
              last_col_ref="last_name",
              party_col_ref="party",
              chamber_col_ref="house_or_senate",
              state_col_ref="state",
              start_col_ref="start_year",
              end_col_ref="end_year",
              elec_name_col="name",
              elec_url_col="url",
              elec_party_col="party",
              elec_year_col="year",
              verbose=True):

    # collapse datasets
    ref = ref.copy()
    elecs = elecs.copy()
    ref_grp = ( # congress memebers dataset
        ref.groupby([first_col_ref, last_col_ref, state_col_ref, party_col_ref], dropna=False)
           .agg(**{
               name_col_ref: (name_col_ref, "first"),
               chamber_col_ref: (chamber_col_ref, lambda x: ",".join(sorted(set(map(str, x))))),
               start_col_ref: (start_col_ref, "min"),
               end_col_ref: (end_col_ref, "max"),
           })
           .reset_index()
    )
    ref_grp["_ref_id"] = np.arange(len(ref_grp))
    eb = elecs[[elec_url_col, elec_name_col, elec_party_col, elec_year_col]].copy()
    eb[elec_party_col] = eb[elec_party_col].astype(str)
    eb[elec_year_col] = eb[elec_year_col].astype(str)
    elec_grp = ( # elections datasets
        eb.groupby([elec_url_col, elec_name_col], dropna=False)
          .agg(
              party=(elec_party_col, lambda s: ", ".join(sorted({p.strip() for p in s if p.strip()}))),
              election_years=(elec_year_col, lambda x: ", ".join(sorted(set(x)))),
          )
          .reset_index()
    )
    elec_grp["_elec_id"] = np.arange(len(elec_grp))

    # precompute tokens and features
    ref_grp["_ref_name_norm"] = ref_grp[name_col_ref].apply(normalize_text)
    ref_grp["_ref_words"] = ref_grp["_ref_name_norm"].apply(str.split)
    ref_grp["_ref_suffix"] = ref_grp["_ref_words"].apply(extract_suffix)
    ref_grp["_ref_first_primary"] = ref_grp[first_col_ref].apply(first_token_keep_noninitials)
    ref_grp["_ref_first_opts"] = ref_grp["_ref_first_primary"].apply(lambda t: set(expand_firstname_options(t)))
    ref_grp["_ref_first_pool"] = ref_grp.apply(lambda r: set(firstname_variant_pool(r[first_col_ref], r[name_col_ref], r[last_col_ref])), axis=1)
    ref_grp["_ref_mids"] = ref_grp[first_col_ref].apply(middle_initials_from_first)
    ref_grp["_ref_last_tokens"] = ref_grp[last_col_ref].apply(surname_tokens_from_last)
    ref_grp["_ref_last_tokens_set"] = ref_grp["_ref_last_tokens"].apply(set)
    ref_grp["_ref_party"] = ref_grp[party_col_ref].apply(lambda x: str(x).strip().lower() if pd.notna(x) else "")
    r_expl = ref_grp[["_ref_id"]].join(ref_grp["_ref_last_tokens"].explode().rename("_ln_tok"))
    elec_grp["_elec_name_norm"] = elec_grp[elec_name_col].apply(normalize_text)
    elec_grp["_elec_words"] = elec_grp["_elec_name_norm"].apply(str.split)
    elec_grp["_elec_suffix"] = elec_grp["_elec_words"].apply(extract_suffix)
    elec_grp["_elec_tokens_set"] = elec_grp["_elec_words"].apply(set)
    elec_grp["_elec_last_tokens"] = elec_grp[elec_name_col].apply(surname_tokens_from_fullname)
    elec_grp["_elec_last_set"] = elec_grp["_elec_last_tokens"].apply(set)
    elec_grp["_elec_party"] = elec_grp["party"].apply(lambda x: str(x).strip().lower() if pd.notna(x) else "")
    e_expl = elec_grp[["_elec_id"]].join(elec_grp["_elec_last_tokens"].explode().rename("_ln_tok"))

    # surname merge and flag for no-lastname and fail-lastname
    cand = r_expl.merge(e_expl, on="_ln_tok", how="left").dropna(subset=["_elec_id"])
    cand["_elec_id"] = cand["_elec_id"].astype(int)
    nolast_mask = ref_grp[last_col_ref].fillna("").astype(str).str.strip().eq("")
    nolast_ids = set(ref_grp.loc[nolast_mask, "_ref_id"])
    have_any = set(cand["_ref_id"].unique())
    all_ids = set(ref_grp["_ref_id"])
    fail_last_ids = (all_ids - have_any) - nolast_ids
    cand = cand[["_ref_id","_elec_id"]].drop_duplicates()

    # attach features to candidates
    ref_base = ref_grp.set_index("_ref_id")
    elec_base = elec_grp.set_index("_elec_id")
    def pick(series, key): return series.map(key)

    cand["_ref_first_opts"] = pick(cand["_ref_id"], ref_base["_ref_first_opts"])
    cand["_ref_first_pool"] = pick(cand["_ref_id"], ref_base["_ref_first_pool"])
    cand["_ref_last_set"] = pick(cand["_ref_id"], ref_base["_ref_last_tokens_set"])
    cand["_ref_words"] = pick(cand["_ref_id"], ref_base["_ref_words"])
    cand["_ref_suffix"] = pick(cand["_ref_id"], ref_base["_ref_suffix"])
    cand["_ref_mids"] = pick(cand["_ref_id"], ref_base["_ref_mids"])
    cand["_ref_party"] = pick(cand["_ref_id"], ref_base["_ref_party"])

    cand["_elec_tokens"] = pick(cand["_elec_id"], elec_base["_elec_tokens_set"])
    cand["_elec_last_set"] = pick(cand["_elec_id"], elec_base["_elec_last_set"])
    cand["_elec_words"] = pick(cand["_elec_id"], elec_base["_elec_words"])
    cand["_elec_suffix"] = pick(cand["_elec_id"], elec_base["_elec_suffix"])
    cand["_elec_party"] = pick(cand["_elec_id"], elec_base["_elec_party"])

    # boolean filters
    cand["first_primary"] = cand.apply(lambda r: len(r["_ref_first_opts"] & r["_elec_tokens"]) > 0, axis=1)
    cand["first_fallback"] = cand.apply(lambda r: (not r["first_primary"]) and len(r["_ref_first_pool"] & r["_elec_tokens"]) > 0, axis=1)
    #cand["party_hit"] = (cand["_ref_party"] != "") & cand.apply(lambda r: (r["_ref_party"] in r["_elec_party"]) or (r["_elec_party"] in r["_ref_party"]), axis=1) # name-based matching only
    cand["first_token_match"] = cand["_elec_words"].apply(lambda w: bool(w)) & cand.apply(lambda r: r["_elec_words"][0] in r["_ref_first_opts"], axis=1)
    cand["last_token_match"] = cand.apply(lambda r: len(r["_ref_last_set"])>0 and r["_ref_last_set"].issubset(r["_elec_last_set"]), axis=1)

    def mids_hit(mids, words):
        if not mids: return False
        inner = words[1:-1] if len(words) > 2 else []
        return any(w and w[0] in mids for w in inner)
    cand["initials_match"] = cand.apply(lambda r: mids_hit(r["_ref_mids"], r["_elec_words"]), axis=1)

    def suffix_overlap(ref_suffix, elec_words):
        if not ref_suffix: return False
        return len(ref_suffix & set(elec_words[-3:])) > 0
    cand["suffix_match"] = cand.apply(lambda r: suffix_overlap(r["_ref_suffix"], r["_elec_words"]), axis=1)

    # matching loop
    def try_filter(df, col):
        if df.empty: return df
        sub = df[df[col]]
        return sub if not sub.empty else df

    rows = []
    elec_view = elec_base.reset_index()[["_elec_id", elec_name_col, elec_url_col]].rename(columns={elec_name_col:"name", elec_url_col:"url"})

    # iterate
    for _ref_id in ref_grp["_ref_id"]:
        r = ref_base.loc[_ref_id]
        grp_cand = cand[cand["_ref_id"] == _ref_id]

        # nolastname
        if _ref_id in nolast_ids:
            if verbose: print(f"[nolastname] No last name for {r[name_col_ref]}")
            rows.append({**{ "ref_name":r[name_col_ref], "ref_first":r[first_col_ref], "ref_last":r[last_col_ref],
                             "elec_name":None,"elec_url":None, chamber_col_ref:r[chamber_col_ref],
                             state_col_ref:r[state_col_ref], party_col_ref:r[party_col_ref],
                             start_col_ref:r[start_col_ref], end_col_ref:r[end_col_ref]}, "match":"nolastname"})
            continue

        # fail-lastname
        if _ref_id in fail_last_ids:
            if verbose: print(f"[fail-lastname] {r[name_col_ref]}: no candidates with last name '{str(r[last_col_ref]).strip()}'")
            rows.append({**{ "ref_name":r[name_col_ref], "ref_first":r[first_col_ref], "ref_last":r[last_col_ref],
                             "elec_name":None,"elec_url":None, chamber_col_ref:r[chamber_col_ref],
                             state_col_ref:r[state_col_ref], party_col_ref:r[party_col_ref],
                             start_col_ref:r[start_col_ref], end_col_ref:r[end_col_ref]}, "match":"fail-lastname"})
            continue

        step = grp_cand[grp_cand["first_primary"]]
        if step.empty:
            step = grp_cand[grp_cand["first_fallback"]]

        if step.empty:
            # only-surname
            if len(grp_cand) == 1:
                if verbose:
                    print(f"[only-surname] {r[name_col_ref]}: one candidate with last name '{str(r[last_col_ref]).strip()}' but firstname mismatch")
                    print(elec_view[elec_view["_elec_id"].isin(grp_cand["_elec_id"])][["name","url"]].to_string(index=False))
                match = "only-surname"
            # fail-firstname
            else:
                if verbose:
                    ft = first_token_keep_noninitials(r[first_col_ref])
                    opts = [o.capitalize() for o in expand_firstname_options(ft)] if ft else []
                    msg = f"[fail-firstname] {r[name_col_ref]}: no candidates with last name '{str(r[last_col_ref]).strip()}'"
                    print(msg + (f" and first name '{', '.join(opts)}'" if opts else " (no usable tokens)"))
                match = "fail-firstname"

            rows.append({**{ "ref_name":r[name_col_ref], "ref_first":r[first_col_ref], "ref_last":r[last_col_ref],
                             "elec_name":None,"elec_url":None, chamber_col_ref:r[chamber_col_ref],
                             state_col_ref:r[state_col_ref], party_col_ref:r[party_col_ref],
                             start_col_ref:r[start_col_ref], end_col_ref:r[end_col_ref]}, "match":match})
            continue

        for col in [
            #"party_hit", # name-based matching only
            "first_token_match","last_token_match","initials_match","suffix_match"]:
            step = try_filter(step, col)

        if len(step) > 1:
            ref_last_tokens = list(r["_ref_last_tokens_set"])
            ref_last = ref_last_tokens[-1] if ref_last_tokens else None
            if ref_last:
                def last_non_suffix(words):
                    for w in reversed(words):
                        if w not in SUFFIX_TOKENS:
                            return w
                    return None
                step_last = step[step["_elec_words"].apply(lambda w: last_non_suffix(w) == ref_last)]
                if not step_last.empty and len(step_last) < len(step):
                    step = step_last

        if len(step) == 1:
            # success
            e = elec_base.loc[int(step["_elec_id"].iloc[0])]
            rows.append({**{ "ref_name":r[name_col_ref], "ref_first":r[first_col_ref], "ref_last":r[last_col_ref],
                             "elec_name":e[elec_name_col], "elec_url":e[elec_url_col],
                             chamber_col_ref:r[chamber_col_ref], state_col_ref:r[state_col_ref],
                             party_col_ref:r[party_col_ref], start_col_ref:r[start_col_ref],
                             end_col_ref:r[end_col_ref]}, "match":"success"})
        else:
            # ambiguous
            if verbose:
                print(f"[ambiguous] {r[name_col_ref]}: multiple candidates remain after deterministic filters:")
                print(elec_view[elec_view["_elec_id"].isin(step["_elec_id"])][["name","url"]].to_string(index=False))
            rows.append({**{ "ref_name":r[name_col_ref], "ref_first":r[first_col_ref], "ref_last":r[last_col_ref],
                             "elec_name":None,"elec_url":None, chamber_col_ref:r[chamber_col_ref],
                             state_col_ref:r[state_col_ref], party_col_ref:r[party_col_ref],
                             start_col_ref:r[start_col_ref], end_col_ref:r[end_col_ref]}, "match":"ambiguous"})

    df = pd.DataFrame(rows, columns=["ref_name","ref_first","ref_last","elec_name","elec_url",
                                     chamber_col_ref,state_col_ref,party_col_ref,start_col_ref,end_col_ref,"match"])
    mat = df[df["match"]=="success"].copy().reset_index(drop=True)
    unmat = df[df["match"]!="success"].copy().reset_index(drop=True)
    return mat, unmat
