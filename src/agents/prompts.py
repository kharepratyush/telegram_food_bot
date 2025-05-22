"""
This module defines the core system prompt for the Food Agent and provides
functionality to dynamically update it with recent food log data.
"""

from datetime import datetime

from src.agents.utils.db import get_recent_food_log

# FOOD_PROMPT: Core System Prompt for the Food Agent
#
# Purpose:
# This prompt guides the LLM to act as a specialized food assistant. It sets
# the context, capabilities, and constraints for the agent's responses.
#
# Key Constraints & Instructions for the LLM:
# - Specializes in diabetic-friendly meal planning for pregnant individuals,
#   with a focus on North Indian cuisine.
# - Must always create a meal plan when requested.
# - Can use internet search for details if necessary.
# - Responses should be in Markdown and include:
#   - Detailed macro- and micronutrient breakdowns.
#   - Evidence-based insights (though currently restricted to only provide meal plan).
#   - Clear guidance on glycemic index (GI) values.
#   - Culturally relevant meal/snack ideas and preparation tips.
# - Strict dietary rule: No non-vegetarian options on Tuesdays and Thursdays.
# - Output Restriction: Only provide the meal plan itself, without extra information
#   or self-care insights, unless specifically part of the meal plan request.
# - Meal Plan Structure: Each meal should include Carbs and Protein components,
#   with a preference for protein-heavy meals (e.g., chicken, where appropriate).
# - Output Format: Must include a JSON block with specified headers and rows
#   for easy parsing, alongside textual recommendations.
#
# Usage:
# This prompt is a foundational part of the LangGraph agent's configuration.
# The `update_prompt` function dynamically augments this base prompt by
# prepending information about recently logged food items, instructing the
# LLM to avoid repetition.
FOOD_PROMPT = """
You are a knowledgeable food assistant specializing in diabetic-friendly meal planning, with a focus on North Indian cuisine for pregnant individuals managing diabetes.

Always respond in Markdown:

- Dont reply with meal plan unless explicitly asked.
- Search internet to get details if required
- Detailed macro- and micronutrient breakdowns for each recipe or meal suggestion
- Evidence-based insights from dietitians and endocrinologists on diet, exercise, and self-care during pregnancy
- Clear guidance on glycemic index (GI) values
- Culturally relevant meal and snack ideas and preparation tips
- A strict rule: no non-vegetarian options on Tuesdays and Thursdays
- Only provide exactly what is requested — no extra recommendations or self-care insights
- Only provide the meal plan without any extra information
- Add complete meal plan: Each meal should have Carbs and Protein components.
- Make meal protein heavy, preferably chicken
- Be descriptive & provide as much information as possible

Example Output:


## Recommendations:
**Breakfast**:
- Recipe: Vegetable Upma
- Macronutrients: Calories, Carbs, Protein, Fat
- Micronutrients: Fiber, Iron, Folate
- GI Value: Low (GI: 55)

…

**Dinner**:
- Recipe: Palak Paneer
- Macronutrients: Calories, Carbs, Protein, Fat
- Micronutrients: Calcium, Vitamin A, Vitamin C
- GI Value: Low (GI: 30)

```json
{
  "headers": ["Meal", "Recipe", "Calories", "Carbs (g)", "Protein (g)", "Fat (g)", "Fiber (g)", "GI Value"],
  "rows": [
    ["Breakfast", "Vegetable Upma", "250", "30", "6", "8", "4", "55"],
    …
    ["Dinner", "Palak Paneer", "300", "10", "12", "20", "5", "30"]
  ]
}
```"""


async def update_prompt(base_prompt: str) -> str:
    """Updates the given base prompt with recent food log information.

    This function fetches recent food log entries and prepends them to the
    base prompt, along with an instruction to the LLM not to repeat dishes
    found in the log. If no logs are found, the base prompt is returned unchanged.

    Args:
        base_prompt: The initial prompt string to be updated.

    Returns:
        The updated prompt string with food log information, or the original
        base_prompt if no logs are available or an error occurs.
    """
    try:
        recent_logs = await get_recent_food_log()
    except Exception:
        recent_logs = []

    if recent_logs:
        # Construct the "Don't repeat" message and the log entries string
        dont_repeat_message = "\nDont repeat the dish in food log for future meals."
        log_entries_str = "\n".join(
            f"{datetime.fromtimestamp(float(timestamp))} - {meal}: {dish}"
            for timestamp, meal, dish in recent_logs
        )
        # Prepend the "Don't repeat" message, then the "Recent food log" header, then the logs
        prompt_addition = (
            f"{dont_repeat_message}\n\n" f"Recent food log:\n{log_entries_str}"
        )
        # The problem description for tests implied "Dont repeat" *before* "Recent food log:"
        # and the food log section *appended* to the main prompt.
        # The original code appends `prompt_addition` to `PROMPT`.
        # Let's ensure the structure is: BASE_PROMPT + DONT_REPEAT_MSG + LOG_HEADER + LOGS
        # Current: PROMPT += prompt_addition -> BASE_PROMPT + DONT_REPEAT_MSG + LOG_HEADER + LOGS
        # This means the "Dont repeat" message and logs are added at the end of the base_prompt.
        # The tests for prompts.py confirm this structure.
        return f"{base_prompt}{prompt_addition}"
    return base_prompt
