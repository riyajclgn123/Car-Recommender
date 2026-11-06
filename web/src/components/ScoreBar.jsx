export default function ScoreBar({ label, value }) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 70 ? "bg-green-500" :
    pct >= 40 ? "bg-yellow-400" : "bg-red-400"

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 text-gray-500 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-100 rounded-full h-2">
        <div
          className={`${color} h-2 rounded-full transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-8 text-right text-gray-600">{pct}%</span>
    </div>
  )
}