import ScoreBar from "./ScoreBar"

export default function CarCard({ rec, rank }) {
  const deal = rec.predicted_price - rec.listing_price
  const dealLabel = deal > 500
    ? `$${deal.toLocaleString()} below market`
    : deal < -500
    ? `$${Math.abs(deal).toLocaleString()} above market`
    : "At fair market value"

  return (
    <div className="bg-white rounded-2xl shadow-md p-6 border border-gray-100
                    hover:shadow-lg transition-shadow">
      {/* Header */}
      <div className="flex justify-between items-start mb-3">
        <div>
          <span className="text-xs font-semibold text-indigo-600
                           bg-indigo-50 px-2 py-0.5 rounded-full">
            #{rank}
          </span>
          <h3 className="mt-1 text-lg font-bold text-gray-900">
            {rec.year} {rec.make_name} {rec.model_name}
          </h3>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-gray-900">
            ${rec.listing_price.toLocaleString()}
          </div>
          <div className="text-xs text-gray-400">listing price</div>
        </div>
      </div>

      {/* Price + TCO row */}
      <div className="grid grid-cols-3 gap-3 mb-4 p-3
                      bg-gray-50 rounded-xl text-center">
        <div>
          <div className="text-sm font-semibold text-gray-700">
            ${rec.predicted_price.toLocaleString()}
          </div>
          <div className="text-xs text-gray-400">fair price</div>
        </div>
        <div>
          <div className="text-sm font-semibold text-gray-700">
            ${rec.tco_5yr.toLocaleString()}
          </div>
          <div className="text-xs text-gray-400">5-yr cost</div>
        </div>
        <div>
          <div className="text-sm font-semibold text-gray-700">
            ${rec.monthly_cost.toLocaleString()}/mo
          </div>
          <div className="text-xs text-gray-400">monthly</div>
        </div>
      </div>

      {/* Deal badge */}
      <div className={`text-xs font-medium mb-3 px-2 py-1 rounded-lg
                       inline-block ${
        deal > 500 ? "bg-green-50 text-green-700" :
        deal < -500 ? "bg-red-50 text-red-600" :
        "bg-gray-50 text-gray-600"
      }`}>
        {dealLabel}
      </div>

      {/* Score bars */}
      <div className="space-y-1.5 mb-4">
        <ScoreBar label="TCO fit"     value={rec.score_breakdown.tco_fit} />
        <ScoreBar label="Deal"        value={rec.score_breakdown.deal} />
        <ScoreBar label="Reliability" value={rec.score_breakdown.reliability} />
        <ScoreBar label="Life-stage"  value={rec.score_breakdown.lifestage} />
        <ScoreBar label="Climate"     value={rec.score_breakdown.climate} />
      </div>

      {/* Why this car */}
      <div className="border-t pt-3">
        <p className="text-xs text-gray-500 leading-relaxed italic">
          {rec.why}
        </p>
      </div>
    </div>
  )
}