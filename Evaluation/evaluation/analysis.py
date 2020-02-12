from collections import defaultdict
from sympy.polys.polytools import poly
from maccabee.constants import Constants

def get_term_category_data(dgps):
    sampled_dgp_transforms = map(lambda x: x[1].treatment_covariate_transforms, dgps)
    category_data = list(map(categorize_terms, sampled_dgp_transforms))
    return category_data

def categorize_interaction_two_way(is_binary_statuses):
    if is_binary_statuses[0] and is_binary_statuses[1]:
        return "both"
    elif not is_binary_statuses[0] and not is_binary_statuses[1]:
        return "neither"
    else:
        return "one"

def categorize_terms(terms):
    counts = defaultdict(int)
    components = defaultdict(list)

    for term in terms:
        if not term.free_symbols:
            continue

        poly_term = poly(term)
        variables = poly_term.gens
        max_deg = max(poly_term.degree_list())

        if len(variables) == 2:
            assert(max_deg == 1)
            category = Constants.DGPSampling.INTERACTION_TWO_WAY
        elif len(variables) == 3:
            assert(max_deg == 1)
            category = Constants.DGPSampling.INTERACTION_THREE_WAY
        elif len(variables) > 3:
            raise Exception("Unexpected degree!")
        elif max_deg == 2:
            assert(len(variables) == 1)
            category = Constants.DGPSampling.POLY_QUADRATIC
        elif max_deg == 3:
            assert(len(variables) == 1)
            category = Constants.DGPSampling.POLY_CUBIC
        elif not variables[0].is_Atom:
            assert(len(variables) == 1)
            category = STEP_CONSTANT + "/" + STEP_VARIABLE
        else:
            assert(len(variables) ==1 and max_deg == 1)
            category = Constants.DGPSampling.LINEAR

        counts[category] += 1
        components[category].append((variables))

    return counts, components
