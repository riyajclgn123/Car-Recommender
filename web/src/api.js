import axios from "axios"

const BASE = "http://localhost:8080"

export const predictPrice  = (data) =>
  axios.post(`${BASE}/predict-price`, data).then(r => r.data)

export const getTCO = (data) =>
  axios.post(`${BASE}/tco`, data).then(r => r.data)

export const getRecommendations = (data) =>
  axios.post(`${BASE}/recommend`, data).then(r => r.data)

export const parseProfileWithClaude = async (freeText) => {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 1000,
      messages: [{
        role: "user",
        content: `Extract a car buyer profile from this text and return ONLY valid JSON.
No explanation, no markdown, just the JSON object.

Text: "${freeText}"

Return this exact structure (fill in best guesses for missing fields):
{
  "monthly_income": <number, monthly gross income in USD>,
  "max_maintenance_monthly": <number, default 150>,
  "household_size": <integer 1-6>,
  "age_group": <"18-34" | "35-54" | "55-64" | "65+">,
  "income_band": <"low" | "middle" | "high">,
  "area_type": <"urban" | "suburban" | "rural">,
  "state": <2-letter US state code, default "TX">,
  "prefer_new": <true | false | null>
}

Rules:
- monthly_income: if annual given, divide by 12
- income_band: low=under $40k/yr, middle=$40-100k, high=over $100k
- If not mentioned, use sensible defaults for a typical US car buyer`
      }]
    })
  })
  const data = await res.json()
  const text = data.content[0].text.trim()
  return JSON.parse(text)
}