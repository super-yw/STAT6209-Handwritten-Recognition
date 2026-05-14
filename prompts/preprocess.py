CHECK_PROMPT = """You are an invoice layout analyzer. Your task is to detect whether this invoice contains an **extra content section below the main item table**.

    ## Definitions:
    - **Main item table**: The primary structured table containing product/item rows with columns such as item code, description, quantity, unit price, amount, etc. This table ends at its bottom border line (including any "Total" summary row).
    - **Extra section**: Any additional structured content (text blocks, sub-tables, form fields, or boxes) that appears **below the bottom border of the main item table**, separated visually from it.

    ## Instructions:
    1. Locate the bottom edge of the main item table (including its total/subtotal row).
    2. Check if there is any additional content region below that bottom edge (not just blank space or a simple footer signature line).
    3. If such an extra section exists:
    - Output `"has_extra_section": true`
    - Output the bounding box using **normalized relative coordinates** (values between 0.0 and 1.0):
        - `x_min`: left edge of the extra section ÷ image width
        - `y_min`: top edge of the extra section ÷ image height
        - `x_max`: right edge of the extra section ÷ image width
        - `y_max`: bottom edge of the extra section ÷ image height
    - Format: `"bbox_relative": [x_min, y_min, x_max, y_max]`
    4. If no such extra section exists:
    - Output `"has_extra_section": false`
    - Output `"bbox_relative": null`

    ## Coordinate rules:
    - Origin (0.0, 0.0) is the **top-left corner** of the image
    - (1.0, 1.0) is the **bottom-right corner** of the image
    - All values must be **float numbers between 0.0 and 1.0**
    - Example: a box covering the bottom 30% of the image, full width → [0.0, 0.7, 1.0, 1.0]

    ## Output format (strict JSON, no extra text):
    {
    "has_extra_section": <true | false>,
    "bbox_relative": [x_min, y_min, x_max, y_max] or null,
    "reasoning": "<brief explanation of what you found>"
    }

    ## Important notes:
    - Only consider content that forms a **structured block** (has visible borders, form fields, or organized text layout) as an extra section.
    - Simple single-line footers (e.g., page numbers, stamps, signature lines) do NOT count as extra sections.
    - The extra section typically has its own visible bounding box/border drawn around it.

    """

FIND_ADDON_MODIF_PROMPT = """The image shows an invoice.

        Find out which item has a handwritten number value (on the right of the item name).

        The return MUST be in the format: [{"item name": [value]}, {"item name": [value]}].
        
        If neither has a handwritten value, return [].

        Follow the following steps:
        1. Find out how many handwritten numbers are in the picture. If no handwritten numbers, skip the following steps and return [] derectly.
        2. Indentify what are these handwritten numbers.
        3. Identify which item each handwritten number belongs to.
        4. Return in the required format: [{"item name": [value]}, {"item name": [value]}].
        """
