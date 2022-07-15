"""
This module include the basic objects of a graph: Nodes and Edges.

Example on how to use them under the __main__ flag. 
"""


class BaseGraphObject(dict):
    """
    Base class for individual Graph objects: Nodes/Edges
    """

    def __init__(self, *args, **kwargs):
        super(BaseGraphObject, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def _construct_from_dict(cls, basic_attrib, properties_dict):
        """
        Construc object from a dictionaries of basic attributes (~id, ~label, etc.) and one of properties where keys are attriute names (key) and values are attribue values.

        Args:
            basic_attrib (dict): Dictionary with basic attribues. Either one with ~id and ~label for node or ~id, ~from, ~to and ~label for edges.\n
            properties_dict (dict): Any additional properties that the node or edge may have.\n

        Returns:
            baseGraphObject: The graph node object.
        """
        basic_attrib.update(**properties_dict)
        obj = cls(**basic_attrib)
        return obj

    def get_prop_value(self, p):
        """
        Retrieve the property of the node as string. If it's:
            -nan
            -NaT
            -Unknown (not case sensitive)
        it will be replaced by an empty string ''.

        Args:
            p (str): Property to check.

        Returns:
            -: Value of the property in the dictionary.
        """
        v = str(self.__dict__[p])
        if v is None or v == "nan" or v == "NaT" or v.lower() == "unknown":
            v = "unknown"

        return str(v)

    def get_dict(self):
        """
        Given a set of properties list it will create a csv line in the same order provided by the properties with it's values.

        Args:
            properties (list): List of properties.

        Returns:
            str: CSV line with properties' values.
        """
        return self.__dict__


class Node(BaseGraphObject):
    """
    Graph's individual node object
    """

    def __init__(self, *args, **kwargs):
        super(Node, self).__init__(*args, **kwargs)

    @classmethod
    def construct_from_dict(cls, node_label, uid, **properties_dict):
        """
        Construct the node object from a dictionary.

        Args:
            label (str): Label of the node.\n
            uid (str): Unique identifier of the node.\n
            properties_dict (dict): any additional properties of the nodes may be passed as kwargs.\n

        Returns:
            node: Node object with input parameters.
        """
        node_attributes = {"~id": uid, "~label": node_label}
        node = cls._construct_from_dict(node_attributes, properties_dict)
        return node


class Edge(BaseGraphObject):
    """
    Graph's individual edge object
    """

    def __init__(self, *args, **kwargs):
        super(Edge, self).__init__(*args, **kwargs)

    @classmethod
    def construct_from_dict(
        cls, edge_label, from_node, to_node, uid, **properties_dict
    ):
        """
        Construct the edge object from a dictionary.
        Args:
            label (str): Label of the edge.\n
            from_node (str): uid of the "from"/"starting" node of the edge.\n
            to_node (str): uid of the "to"/"ending" node of the edge.\n
            uid (str): Unique identifier of the edge.\n
            properties_dict (dict): any additional properties of the nodes may be passed as kwargs.\n

        Returns:
            edge: Edge object with input parameters.
        """
        node_attributes = {
            "~from": from_node,
            "~to": to_node,
            "~id": uid,
            "~label": edge_label,
        }
        node = cls._construct_from_dict(node_attributes, properties_dict)
        return node


# Usage example
if __name__ == "__main__":

    # A node dict **must** contain a *uid*
    label = "test_node"
    node_prop_example = {
        "uid": 134,
        "prop_bool": True,
        "prop_int": 1,
        "prop_str": "test",
        "prop_float": 0.35,
    }

    test_node = Node.construct_from_dict(label, **node_prop_example)

    # An edge dict **must** contain a *uid*, *from_node*, *to_node*.
    edge_prop_example = {
        "uid": 134,
        "from_node": 123,
        "to_node": 345,
        "prop_bool": True,
        "prop_int": 1,
        "prop_str": "test",
        "prop_float": 0.35,
    }

    test_edge = Edge.construct_from_dict(label, **edge_prop_example)
