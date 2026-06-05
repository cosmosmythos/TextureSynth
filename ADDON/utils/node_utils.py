"""Node utilities for node positioning and frame management in TextureSynth."""


def recursive_framed_location_finder(node, loc_xy):
    """Recursively calculate the absolute location of a node accounting for parent frames."""
    locx, locy = loc_xy
    if node.parent:
        locx += node.parent.location.x
        locy += node.parent.location.y
        return recursive_framed_location_finder(node.parent, (locx, locy))
    else:
        return locx, locy


def offset_node_location(source_node, new_node, offset):
    """Position a new node to the right of a source node at a specified offset."""
    new_node.location = (
        source_node.location.x + offset[0] + source_node.width,
        source_node.location.y + offset[1]
    )


def frame_adjust(source_node, new_node):
    """Place the new node in the same parent frame as the source node."""
    if source_node.parent:
        new_node.parent = source_node.parent
        loc_xy = new_node.location[:]
        locx, locy = recursive_framed_location_finder(new_node, loc_xy)
        new_node.location = locx, locy
