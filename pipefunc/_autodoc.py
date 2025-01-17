from rich.markdown import Markdown


def format_pipeline_docs(
    descriptions: dict[str, str],
    returns: dict[str, str],
    parameters: dict[str, list[str]],
) -> str:
    """Formats pipeline documentation using rich for better readability.

    Parameters
    ----------
    descriptions
        Dictionary of descriptions for each output.
    returns
        Dictionary of return descriptions for each output.
    parameters
        Dictionary of parameter descriptions for each function.

    Returns
    -------
        Formatted string containing the documentation.

    """
    doc = ""

    # Function Descriptions
    if descriptions:
        doc += "# Function Descriptions\n"
        for output_name, desc in descriptions.items():
            doc += f"## {output_name}\n{desc}\n"

    # Return Values
    if returns:
        doc += "# Return Values\n"
        for output_name, ret_desc in returns.items():
            doc += f"## {output_name}\n{ret_desc}\n"

    # Parameters
    if parameters:
        doc += "# Parameters\n"
        for param, param_descs in parameters.items():
            doc += f"## {param}\n"
            for desc in param_descs:
                doc += f"- {desc}\n"

    markdown = Markdown(doc)
    return markdown


# # Example usage
# descriptions, returns, parameters = pipeline.doc()
# formatted_doc = format_pipeline_docs(descriptions, returns, parameters)
# console = Console()
