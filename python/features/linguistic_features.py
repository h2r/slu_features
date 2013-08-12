

def merge(dict1, dict2):
    #check_dicts(dict1, dict2)
    result = {}
    result.update(dict1)
    result.update(dict2)
    return result


import enchant
enchant.Dict.__del__ = lambda x: None # silencing weird error
enchantDict = enchant.Dict("en_us")


def depluralize(word):
    stems = []
    if(word == 's'):
        return word
    if word[-2:] == 'es':
        stems.append(word[:-2])
    if word[-1:] == 's':
        stems.append(word[:-1])
    for stem in stems:
        if enchantDict.check(stem):
            return stem
    return word

def sfe_language_object(f_words, visible_obj_words, selected_obj_words):
    """
    Compute language-object features for groups of words.
    """

    f_words = [depluralize(w) for w in f_words]
    
    result = {}
    if visible_obj_words != None:
        visible_obj_words = [depluralize(w) for w in visible_obj_words]
        map1 = word_features(f_words, visible_obj_words, "f_context")
        result = merge(result, map1)
    
    if selected_obj_words != None:
        selected_obj_words = [depluralize(w) for w in selected_obj_words]
        map3 = word_features(f_words, selected_obj_words, "f_selected")
        result = merge(result, map3)

    return result

def word_features(command_words, environment_words, prefix):
    """
    Compute similarity between two groups of words.
    """
    #now iterate through and get the features
    ret_map = {}
    
    if len(command_words) == 0 or len(environment_words) == 0:
        return ret_map
    
    #how much objects are related to a given word in the language
    max_all_flr = 0;     max_all_sim = 0;



    for cword in command_words:
        max_flr = 0; max_sim = 0;
        overlap_cnt = 0

        for eword in environment_words:
            #if the word is in the query
            if eword in cword:
                ret_map[prefix + "_e" + eword + "_in_cword"] = 1.0
                overlap_cnt += 1

            #co-occurrance between two objects
            #ret_names.append(prefix + "_" + cword+"_rel_"+eword)
            #ret_vals.append(1.0)
            ret_map[prefix + "_c_" + cword + "_e_" + eword] = True
            key = prefix + "_c_" + cword + "_e_" + eword

        ret_map[prefix + "_c_" + cword + "_overlap_cnt"] = overlap_cnt
        ret_map[prefix + "_c_" + cword + "_has_overlap"] = True if overlap_cnt != 0 else False
    ret_map = dict((key.replace(" ", "_"), value) for key, value in ret_map.iteritems())

    return ret_map
