'''
gen_util.py

This module contains general functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 2.7.

'''


#############################################
def accepted_values_error(var_name, wrong_value, accept_values):
    """
    accepted_values_error(var_name, wrong_value, accept_values)

    Raises a value error with a message indicating the variable name,
    accepted values for that variable and wrong value stored in the variable.

    Required arguments:
        - var_name (str)      : name of the variable
        - wrong_value (item)  : value stored in the variable
        - accept_values (list): list of accepted values for the variable
    """

    values_str = ', '.join(['\'{}\''.format(x) for x in accept_values])
    error_message = ('\'{}\' value \'{}\' unsupported. Must be in '
                     '{}.').format(var_name, wrong_value, values_str)
    raise ValueError(error_message)


#############################################
def remove_if(vals, rem):
    """
    remove_if(vals, rem)

    Removes items from a list if they are in the list.

    Required arguments:
        - vals (item or list): item or list from which to remove elements
        - rem (item or list) : item or list of items to remove from vals

    Return:
        vals (list): list with items removed.
    """

    if not isinstance(rem, list):
        rem = [rem]
    if not isinstance(vals, list):
        vals = [vals]
    for i in rem:
        if i in vals:
            vals.remove(i)
    return vals


#############################################
def list_if_not(vals):
    """
    list_if_not(vals)

    Converts input into a list if not a list.

    Required arguments:
        - vals (item or list): item or list

    Return:
        vals (list): list version of input.
    """
    
    if not isinstance(vals, list):
        vals = [vals]
    return vals