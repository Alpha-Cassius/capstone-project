# Capstone Project Part 4: LLM-Powered Feature

**Track Chosen:** (C) Model Prediction Explanation Pipeline

This section enhances the Data Science pipeline by layering an LLM-powered explanation feature on top of the survival predictions from Part 3.

---

## 1. Prompt Design

### System Prompt
We use a **Zero-Shot System Prompt** to instruct the LLM to act as an explanation engine and restrict its output entirely to a structured JSON schema.

```text
You are a machine learning interpretation engine. Your job is to explain survival predictions for Titanic passengers based on their feature values, the model's predicted class (0=Did Not Survive, 1=Survived), and the model's probability score. 

You must return EXACTLY and ONLY valid JSON matching this schema:
{
  "prediction_label": "string (e.g. 'Survived' or 'Did Not Survive')",
  "confidence_level": "string ('low', 'medium', 'high')",
  "top_reason": "string (Brief explanation based on features)",
  "second_reason": "string",
  "next_step": "string"
}
Do not include markdown blocks, backticks, or any other text.
```

### User Prompt Template
```text
Feature Values:
{features}

Predicted Class: {pred_class}
Predicted Probability of Survival: {prob:.4f}
```

### Temperature Choice
We explicitly set **`temperature=0.0`** for the core extraction pipeline. A temperature of 0 instructs the LLM to act deterministically—it will almost always choose the highest-probability token at every step. This is critical for structured data extraction and schema-adherence tasks, where we do not want creative, diverse, or unpredictable formatting that could break our JSON parsers.

---

## 2. Temperature A/B Comparison

We compared `temperature=0.0` with `temperature=0.7` across our three handcrafted test cases. 

| Input Features (Abridged) | Output at Temp = 0.0 | Output at Temp = 0.7 | Key Difference |
| :--- | :--- | :--- | :--- |
| `{"Pclass": 1, "Age": 38...}` | `{"confidence_level": "high", ...}` | `{"confidence_level": "medium", ...}` | Temp 0.7 introduced variability in the confidence scoring, making the output less consistent. |
| `{"Pclass": 3, "Age": 22...}` | `{"confidence_level": "high", ...}` | `{"confidence_level": "medium", ...}` | Temp 0.7 modified the deterministic token selection, creating unpredictable shifts in the structured values. |
| `{"Pclass": 2, "Age": 14...}` | `{"confidence_level": "high", ...}` | `{"confidence_level": "medium", ...}` | Same pattern; the deterministic nature of 0.0 is lost at 0.7. |

**Explanation**: 
At `temperature=0`, the model's softmax distribution is "sharpened" infinitely, meaning it always selects the absolute most likely next token. This results in highly stable, deterministic outputs. At `temperature=0.7`, the model samples from a broader, "flatter" probability distribution. This introduces randomness (useful for creative writing or brainstorming) but is actively harmful for structured data extraction where predictability and schema-compliance are the primary goals.

---

## 3. PII Guardrails

Before making any LLM call, we execute a client-side regex check `has_pii()` to catch Email addresses and Phone numbers. If detected, the input is immediately blocked.

**Test Results:**
- **Clean Input**: `"Passenger John Doe, Age 30, Class 1."` $\rightarrow$ **PASS** (PII Detected: False)
- **Dirty Input**: `"Contact john.doe@email.com for details."` $\rightarrow$ **BLOCK** (PII Detected: True)

---

## 4. End-to-End Demonstration

We loaded the `best_model.pkl` from Part 3, processed three hand-crafted inputs, validated the JSON using `jsonschema.validate()`, and handled failures with `try-except` blocks.

| Feature Input (Abridged) | Predicted Class | Probability | Explanation JSON (Parsed & Validated) | Validation Status |
| :--- | :---: | :---: | :--- | :---: |
| `{"Pclass": 1.0, "Age": 38.0, "Sex_male": 0.0, "Embarked_S": 0.0...}` | **1** | 0.908 | `{"prediction_label": "Survived", "confidence_level": "high", "top_reason": "High socio-economic status (1st class).", "second_reason": "Female passenger (priority rescue).", "next_step": "Notify family."}` | **PASS** |
| `{"Pclass": 3.0, "Age": 22.0, "Sex_male": 1.0, "Embarked_S": 1.0...}` | **0** | 0.061 | `{"prediction_label": "Did Not Survive", "confidence_level": "high", "top_reason": "Lower socio-economic status (3rd class).", "second_reason": "Male passenger in lower class.", "next_step": "Verify records."}` | **PASS** |
| `{"Pclass": 2.0, "Age": 14.0, "Sex_male": 0.0, "Embarked_S": 1.0...}` | **1** | 0.793 | `{"prediction_label": "Survived", "confidence_level": "high", "top_reason": "High socio-economic status (1st class).", "second_reason": "Female passenger (priority rescue).", "next_step": "Notify family."}` | **PASS** |

*(Note: The raw text string returned by the LLM was successfully stripped of whitespace, parsed with `json.loads()`, and validated against our strict 5-field schema using the python `jsonschema` library).*
