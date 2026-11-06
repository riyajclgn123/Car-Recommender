import { useState } from "react"
import CarCard from "./components/CarCard"
import { getRecommendations, parseProfileWithClaude } from "./api"

const AGE_GROUPS   = ["18-34", "35-54", "55-64", "65+"]
const INCOME_BANDS = ["low", "middle", "high"]
const AREA_TYPES   = ["urban", "suburban", "rural"]
const US_STATES    = [
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
  "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
  "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
  "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
  "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

const DEFAULT_PROFILE = {
  monthly_income: 4000,
  max_maintenance_monthly: 150,
  household_size: 1,
  age_group: "18-34",
  income_band: "middle",
  area_type: "suburban",
  state: "TX",
  prefer_new: null,
}

export default function App() {
  const [mode,       setMode]       = useState("form") // "form" | "text"
  const [profile,    setProfile]    = useState(DEFAULT_PROFILE)
  const [freeText,   setFreeText]   = useState("")
  const [results,    setResults]    = useState(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)
  const [parsing,    setParsing]    = useState(false)

  const updateProfile = (k, v) =>
    setProfile(p => ({ ...p, [k]: v }))

  const handleParseText = async () => {
    if (!freeText.trim()) return
    setParsing(true)
    setError(null)
    try {
      const parsed = await parseProfileWithClaude(freeText)
      setProfile({ ...DEFAULT_PROFILE, ...parsed })
      setMode("form")
    } catch (e) {
      setError("Couldn't parse that — try filling the form directly.")
    } finally {
      setParsing(false)
    }
  }

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const data = await getRecommendations({
        ...profile,
        n_candidates: 2000,
      })
      setResults(data)
    } catch (e) {
      const msg = e.response?.data?.detail
      setError(msg || "Something went wrong. Is the API running?")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br
                    from-slate-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-6 py-4 flex
                        justify-between items-center">
          <div>
            <h1 className="text-xl font-bold text-gray-900">
              🚗 Car Recommender
            </h1>
            <p className="text-xs text-gray-400">
              AI-powered · Fair prices · 5-year cost
            </p>
          </div>
          <span className="text-xs bg-indigo-100 text-indigo-700
                           px-3 py-1 rounded-full font-medium">
            Powered by LightGBM + Claude
          </span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">

        {/* Mode toggle */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setMode("form")}
            className={`px-4 py-2 rounded-lg text-sm font-medium
                        transition-colors ${
              mode === "form"
                ? "bg-indigo-600 text-white"
                : "bg-white text-gray-600 border border-gray-200"
            }`}
          >
            📋 Fill form
          </button>
          <button
            onClick={() => setMode("text")}
            className={`px-4 py-2 rounded-lg text-sm font-medium
                        transition-colors ${
              mode === "text"
                ? "bg-indigo-600 text-white"
                : "bg-white text-gray-600 border border-gray-200"
            }`}
          >
            💬 Describe yourself
          </button>
        </div>

        {/* Free-text mode */}
        {mode === "text" && (
          <div className="bg-white rounded-2xl shadow-sm border
                          border-gray-100 p-6 mb-6">
            <p className="text-sm text-gray-500 mb-3">
              Tell us about yourself and what you need in a car.
              Claude will fill in the form for you.
            </p>
            <textarea
              className="w-full border border-gray-200 rounded-xl p-3
                         text-sm resize-none focus:ring-2
                         focus:ring-indigo-300 focus:outline-none"
              rows={4}
                            placeholder='e.g. "I am 28, single, make $55k/yr in Austin TX, need reliable used car under $20k"'
              value={freeText}
              onChange={e => setFreeText(e.target.value)}
            />
            <button
              onClick={handleParseText}
              disabled={parsing || !freeText.trim()}
              className="mt-3 px-5 py-2 bg-indigo-600 text-white
                         rounded-lg text-sm font-medium disabled:opacity-50
                         hover:bg-indigo-700 transition-colors"
            >
              {parsing ? "Parsing…" : "✨ Parse with Claude"}
            </button>
          </div>
        )}

        {/* Profile form */}
        <div className="bg-white rounded-2xl shadow-sm border
                        border-gray-100 p-6 mb-6">
          <h2 className="font-semibold text-gray-800 mb-4">
            Your Profile
          </h2>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                Monthly income ($)
              </label>
              <input type="number"
                className="w-full border border-gray-200 rounded-lg
                           px-3 py-2 text-sm"
                value={profile.monthly_income}
                onChange={e => updateProfile(
                  "monthly_income", Number(e.target.value))}
              />
            </div>

            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                Max maintenance ($/mo)
              </label>
              <input type="number"
                className="w-full border border-gray-200 rounded-lg
                           px-3 py-2 text-sm"
                value={profile.max_maintenance_monthly}
                onChange={e => updateProfile(
                  "max_maintenance_monthly", Number(e.target.value))}
              />
            </div>

            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                Household size
              </label>
              <input type="number" min={1} max={10}
                className="w-full border border-gray-200 rounded-lg
                           px-3 py-2 text-sm"
                value={profile.household_size}
                onChange={e => updateProfile(
                  "household_size", Number(e.target.value))}
              />
            </div>

            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                Age group
              </label>
              <select
                className="w-full border border-gray-200 rounded-lg
                           px-3 py-2 text-sm"
                value={profile.age_group}
                onChange={e => updateProfile("age_group", e.target.value)}
              >
                {AGE_GROUPS.map(a => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                Income band
              </label>
              <select
                className="w-full border border-gray-200 rounded-lg
                           px-3 py-2 text-sm"
                value={profile.income_band}
                onChange={e => updateProfile("income_band", e.target.value)}
              >
                {INCOME_BANDS.map(b => (
                  <option key={b} value={b}>{b}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                Area type
              </label>
              <select
                className="w-full border border-gray-200 rounded-lg
                           px-3 py-2 text-sm"
                value={profile.area_type}
                onChange={e => updateProfile("area_type", e.target.value)}
              >
                {AREA_TYPES.map(a => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                State
              </label>
              <select
                className="w-full border border-gray-200 rounded-lg
                           px-3 py-2 text-sm"
                value={profile.state}
                onChange={e => updateProfile("state", e.target.value)}
              >
                {US_STATES.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                New or used?
              </label>
              <select
                className="w-full border border-gray-200 rounded-lg
                           px-3 py-2 text-sm"
                value={
                  profile.prefer_new === true  ? "new"  :
                  profile.prefer_new === false ? "used" : "either"
                }
                onChange={e => updateProfile("prefer_new",
                  e.target.value === "new"  ? true  :
                  e.target.value === "used" ? false : null
                )}
              >
                <option value="either">Either</option>
                <option value="new">New only</option>
                <option value="used">Used only</option>
              </select>
            </div>
          </div>

          {/* Disclaimer */}
          <p className="text-xs text-gray-400 mt-4">
            This tool provides guidance only, not financial advice.
          </p>
        </div>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full py-3 bg-indigo-600 text-white rounded-xl
                     font-semibold text-sm disabled:opacity-50
                     hover:bg-indigo-700 transition-colors mb-6"
        >
          {loading ? "Finding your top 5…" : "🔍 Find My Cars"}
        </button>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700
                          rounded-xl p-4 mb-6 text-sm">
            {error}
          </div>
        )}

        {/* Results */}
        {results && (
          <div>
            <div className="flex justify-between items-center mb-4">
              <h2 className="font-bold text-gray-900 text-lg">
                Your Top 5
              </h2>
              <span className="text-xs text-gray-400">
                Budget: ${results.budget_max_price.toLocaleString()} max
                · ${results.budget_monthly.toLocaleString()}/mo
              </span>
            </div>
            <div className="space-y-4">
              {results.recommendations.map(rec => (
                <CarCard key={rec.rank} rec={rec} rank={rec.rank} />
              ))}
            </div>
            <p className="text-center text-xs text-gray-400 mt-6">
              All results are within your budget.
              Training data: 2019–2020 listings.
              Apply a ~20–40% upward adjustment for current market prices.
            </p>
          </div>
        )}
      </main>
    </div>
  )
}