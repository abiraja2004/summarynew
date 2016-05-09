from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Tree(object):
    """
    Tree obj.

    Example:

            (3 (2 make) (2 (2 it) (2 human)))
            |_(2 make)
            |_(2 (2 it) (2 human))
                |_(2 it)
                |_(2 human)
    """

    def __init__(self, text, parent=None):
        self.label = None  # sentiment label
        self.p = None
        self.word = None  # word
        self.softmax = None
        self.parent = parent  # reference to parent
        self.subtrees = []  # reference to children
        self.parse(text.strip())

    def count_child(self):
        """Count number of child nodes."""
        return len(self.subtrees)

    def is_leaf(self):
        """Check if current node is a leave."""
        return len(self.subtrees) == 0

    def is_root(self):
        """Check if current node is ROOT."""
        return self.parent is None

    def parse(self, tokens):
        #         import pdb
        """Parse tree from tokens
        :param tokens: string of the tree
        """
        depth = 0
        for i in range(1, len(tokens)):
            char = tokens[i]
            if char == '(':
                depth += 1
                if depth == 1:
                    subtree = Tree(tokens[i:])
                    self.subtrees.append(subtree)
            elif char == ')':
                depth -= 1
                if len(self.subtrees) == 0:
                    pos = tokens.find(' ')
                    self.word = tokens[pos + 1:i]

            if depth == 0 and char == ' ' and self.label is None:
                self.label = tokens[1:i]

            if depth < 0:
                break

    def __repr__(self):
        """
        Print tree.

        :return:

                (3 (2 make) (2 (2 it) (2 human)))
        """
        ans = ''
        ans += '(' + self.label

        if self.word is not None:
            ans += ' ' + self.word

        for i in range(len(self.subtrees)):
            ans += ' ' + repr(self.subtrees[i])

        ans += ')'

        return "{}".format(ans)

    def show_tag(self):
        """Print all tags."""
        ans = self.label

        for i in range(len(self.subtrees)):
            ans += ' ' + (self.subtrees[i].show_tag())

        return "{}".format(ans)

    def left_traverse(self, fn, args=None):
        """
        Traverse leaves from left to right.

        :param fn:   function on tree
        :param args:   arguments for the function
        """
        fn(self, args)
        for subtree in self.subtrees:
            subtree.left_traverse(fn, args)

    def to_sentence(self):
        """
        Return raw sentence.

        Example:

                make it human .
        """

        ans = ''

        if self.word is not None:
            ans = self.word
        else:
            for subtree in self.subtrees:
                words = subtree.to_sentence()
                ans += words + ' '

        return ans.strip()

def merge_bin_tree(tree):
    if tree.is_leaf():
        return tree
    if len(tree.subtrees) == 1:
        tree = tree.subtrees[0]
        return merge_bin_tree(tree)
    elif len(tree.subtrees) == 2:
        left = merge_bin_tree(tree.subtrees[0])
        right = merge_bin_tree(tree.subtrees[1])
        tree.subtrees[0] = left
        tree.subtrees[1] = right
        return tree


def merge_only_child(tree, args=None):
    """

    Args:
        tree: current tree
        args: function pattern - honestly not necessary
    Returns:

    """
    if tree.count_child() == 1:
        tree.label += '_%s' % tree.subtrees[0].label
        tree.word = tree.subtrees[0].word
        tree.subtrees = tree.subtrees[0].subtrees
        for subtree in tree.subtrees:
            subtree.parent = tree



def binarize_fully(infile, outfile):
    """

    Args:
        infile: file contains binary trees
        outfile: file contains fully binary trees

    Returns:

    """
    w = open(outfile, 'w')
    with open(infile, 'r') as f:
        for l in f:
            t = Tree(l)
            t.left_traverse(merge_only_child)
            w.write("%s\n" % str(t))
    w.close()

def get_parenthese_pattern(s):
    """

    Args:
        s: tree format
            (NP (DT No) (NN one))

    Returns:
        Parentheses pattern to compare trees
            (()())

    """
    import re
    pattern = re.compile('[^\(\)]+')
    return pattern.sub('', s)

import nltk

if __name__ == '__main__':

    st = "(ROOT (S (SBAR (IN unless) (S (NP (EX there)) (VP (VBP are) (NP (NP (VBG zoning) (NNS ordinances)) (SBAR (S (VP (TO to) (VP (@VP (VB protect) (NP (PRP$ your) (NN community))) (PP (IN from) (NP (DT the) (@NP (JJS dullest) (@NP (NN science) (NN fiction))))))))))))) (@S (, ,) (@S (NP (NN impostor)) (@S (VP (VBZ is) (VP (@VP (VBG opening) (NP (NN today))) (PP (IN at) (NP (NP (DT a) (NN theater)) (PP (IN near) (NP (PRP you))))))) (. .))))))"
    nltktree = nltk.Tree.fromstring(st)
    nltktree.pretty_print()

    t = Tree(st)
    print (t.to_sentence())
    print (t)
    t = merge_bin_tree(t)
    print(t)

    nltktree = nltk.Tree.fromstring(t.__repr__())
    nltktree.pretty_print()
    # print(t.show_tag())
    # print(get_parenthese_pattern(str(t)))