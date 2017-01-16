import myPCFG
import glob
from nltk.corpus import ptb
from nltk import treetransforms


def get_subtree_overlap(tree1, tree2):
    accuracy = 0

    subtrees1 = []
    subtrees2 = []

    union_count = 0

    for subtree1 in tree1.subtrees():
        # only consider proper subtrees
        if not subtree1.__eq__(tree1):
            subtrees1.append(subtree1)
    for subtree2 in tree2.subtrees():
        # only consider proper subtrees
        if not subtree2.__eq__(tree2):
            subtrees2.append(subtree2)

    subtrees1_size = len(subtrees1)
    subtrees2_size = len(subtrees2)

    for subtree1 in subtrees1:
        if subtree1 in subtrees2:
            union_count += 1
            subtrees2.remove(subtree1)


    accuracy = float(union_count) / (subtrees1_size + subtrees2_size - union_count)
    print accuracy

    return accuracy


# Get the test sections (22) from the Penn Treebank
# THE 2 FILE PATHS HERE NEED TO BE REPLACED WITH WHERE YOUR nltk_data FOLDER IS.
ptb_test_paths = [path.replace("/Users/eriya/nltk_data/corpora/ptb/", "") for path in
                  glob.glob('/Users/eriya/nltk_data/corpora/ptb/WSJ/22/*.MRG')]
ptb_test_parsed_sents = [ptb.parsed_sents(fileids=ptb_test_path) for ptb_test_path in ptb_test_paths]
ptb_test_sents = [ptb.words(fileids=ptb_test_path) for ptb_test_path in ptb_test_paths]


count_total = 0
count_useful = 0
count_correct = 0
accuracy_total = 0

for file in ptb_test_parsed_sents:
    for sentence in file:
        tokens = sentence.leaves()

        # tokens = ["The", "dog", "ran", "well"]

        if len(tokens) < 17: # limit to sentences with length less than 17 tokens
            tree1 = sentence
            treetransforms.chomsky_normal_form(tree1)
            tree2 = myPCFG.get_parse(tokens)

            # Undo CNF transformation
            # treetransforms.un_chomsky_normal_form(tree2)

            print tree1
            print ""
            print tree2

            accuracy_total += get_subtree_overlap(tree1, tree2)


            if tree1.__eq__(tree2):
                print ""
                print "CORRECT!!!!!"
                print ""
                count_correct += 1

            if len(tree2.leaves()) < 2 and len(tree1.leaves()) > 1:
                print "We will ignore this tree"
                count_total += 1
            else:
                count_total += 1
                count_useful += 1

            print ""
            print "Comparable sentences:", count_useful, "Total sentences seen:", count_total
            print ""
            print ""

print "Total count", count_total
print "Count useful", count_useful
print "Correct count", count_correct
print "Accuracy", count_correct / float(count_useful)
print "Subtree accuracy",accuracy_total / float(count_useful)