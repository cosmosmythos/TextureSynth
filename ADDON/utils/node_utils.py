"""
Node utilities for TextureSynth.

Helper functions for node positioning and frame management.
Inspired by Sverchok's sv_node_utils.py
"""


def recursive_framed_location_finder(node, loc_xy):
    """
    Recursively calculate the absolute location of a node accounting for parent frames.
    
    Args:
        node: The node whose location to adjust
        loc_xy: Tuple of (x, y) local coordinates
        
    Returns:
        Tuple of (x, y) absolute coordinates
    """
    locx, locy = loc_xy
    if node.parent:
        locx += node.parent.location.x
        locy += node.parent.location.y
        return recursive_framed_location_finder(node.parent, (locx, locy))
    else:
        return locx, locy


def offset_node_location(source_node, new_node, offset):
    """
    Position a new node relative to a source node.
    
    The new node is positioned to the right of the source node at a specified offset,
    accounting for the source node's width.
    
    Args:
        source_node: The reference node (e.g., the clicked node)
        new_node: The node to position
        offset: Tuple of (x_offset, y_offset) to apply after source position and width
                Example: [100, 250] positions 100px right of source width, 250px up
    """
    new_node.location = (
        source_node.location.x + offset[0] + source_node.width,
        source_node.location.y + offset[1]
    )


def frame_adjust(source_node, new_node):
    """
    Ensure the new node is placed in the same parent frame as the source node.
    
    If the source node is inside a frame, the new node is added to the same frame
    and its local coordinates are adjusted to account for the frame's position.
    
    Args:
        source_node: The reference node (parent frame relationship)
        new_node: The node to adjust
    """
    if source_node.parent:
        new_node.parent = source_node.parent
        loc_xy = new_node.location[:]
        locx, locy = recursive_framed_location_finder(new_node, loc_xy)
        new_node.location = locx, locy
