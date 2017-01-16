from nltk.tree import Tree

token_index = 0
mytokens = []


def get_parse_tree(table, i, j, POS, level):
    psi = table[i][j][POS]

    if psi is 0:
        global token_index
        temp = token_index
        token_index += 1
        global mytokens
        return "(" + POS + " " + mytokens[temp] + ")"

    else:
        i1 = psi[0][0]
        j1 = psi[0][1]
        i2 = psi[1][0]
        j2 = psi[1][1]
        POS1 = psi[2]
        POS2 = psi[3]

        return "(" + POS + " " + get_parse_tree(table, i1, j1, POS1, level+1) + " " + get_parse_tree(table, i2, j2, POS2, level+1) + ")"





# Dictionary that contains counts for each " LHS -> " rule
LHScounts = {}

# Dictionary that contain probability of rules.
# For example, rule_probs["'bank'"] = [('NN', 0.003208050798282408), ('VB', 4.1625041625041625e-05)]
# which means that 'bank' has a 0.003 prob for the rule NN -> 'bank'
#                     and has a 4.16^-5 prob for the rule VB -> 'bank'


rule_probs = {}
rules = {}
terminal_probs_temp = {}
terminal_probs = []

with open('PTB_TRAINING_GRAMMAR_CNF.txt', 'r') as file:
    terminal_count = 0
    terminals = []

    for line in file.readlines():
        LHS = line.split()[1]
        RHS = " ".join(line.split()[3:])
        if LHS in LHScounts:
            LHScounts[LHS] += int(line.split()[0])
        else:
            LHScounts[LHS] = int(line.split()[0])

        # count terminals
        if RHS.startswith("'"):
            terminal_count += 1
            if LHS not in terminal_probs:
                terminal_probs_temp[LHS] = float(line.split()[0])
            else:
                terminal_probs_temp[LHS] += float(line.split()[0])

    # calculate terminal probabilities
    for terminal in terminal_probs_temp:
        terminal_probs.append((terminal, terminal_probs_temp[terminal] / float(terminal_count)))

    file.seek(0)
    for line in file.readlines():
        LHS = line.split()[1]
        RHS = " ".join(line.split()[3:])
        prob = float(line.split()[0]) / LHScounts[LHS]

        if LHS in rules:
            rules[LHS].append((RHS, prob))
        else:
            rules[LHS] = [(RHS, prob)]


        if RHS in rule_probs:
            rule_probs[RHS].append((LHS, prob))
        else:
            rule_probs[RHS] = [(LHS, prob)]

#print rule_probs["'bank'"]



def get_parse(tokens):

    global mytokens, token_index
    token_index = 0
    mytokens = tokens
    #exit()

    #tokens = word_tokenize(sentence)

    for token in tokens:
        token = "'" + token + "'" # The extra apostrophe is required
        try:
            print token, rule_probs[token]
        except KeyError as e:
            # we have an unseen token. Simply give it the probability of terminals
            print "Keyerror:", e
            rule_probs[token] = terminal_probs


    #### PROBABILISTIC CYK PARSING ###
    delta = []
    psi = []
    for i in range(0, len(tokens) +1):
        mydict = dict()
        mydict2 = dict()
        for key in rules.keys(): #populate all cells
            mydict[key] = 0
            mydict2[key] = 0
        some_list = [mydict.copy() for k in range(len(tokens) +1)]
        some_list2 = [mydict2.copy() for k in range(len(tokens) +1)]
        delta.append(some_list)
        psi.append(some_list2)


    # base case
    for i, token in enumerate(tokens):
        for rule_prob in rule_probs["'" + token + "'"]:
            delta[i+1][1][rule_prob[0]] = rule_prob[1]


    # recursive case
    n = len(tokens)
    for length in range(2, n+1):
        for i in range(1, n - length + 1+1):
            for k in range(1, length - 1+1):
                for A in rules.keys():
                    for BCprob in rules[A]: # each item in BCprob is ('B C', prob)
                        RHS = BCprob[0]
                        if not RHS.startswith("'") and len(RHS.split()) > 1: # Okay so for some reason it is not strictly CNF
                            prob = 0
                            B = RHS.split()[0]
                            C = RHS.split()[1]

                            prob = BCprob[1] * delta[i][k][B] * delta[i + k][length - k][C]

                            if prob > delta[i][length][A]:
                                delta[i][length][A] = prob
                                psi[i][length][A] = ((i,k),(i+k,length-k),B,C)


    # return the tree
    return Tree.fromstring(get_parse_tree(psi, 1, len(tokens), "S", 1))