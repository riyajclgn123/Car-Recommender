import axios from "axios"

// Local dev uses 8080; on Vercel set VITE_API_URL to the deployed API URL
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8080"

export const predictPrice = (data) =>
  axios.post(`${BASE}/predict-price`, data).then(r => r.data)

export const getTCO = (data) =>
  axios.post(`${BASE}/tco`, data).then(r => r.data)

export const getRecommendations = (data) =>
  axios.post(`${BASE}/recommend`, data).then(r => r.data)

export const parseProfileFromText = (text) =>
  axios.post(`${BASE}/parse-profile`, { text }).then(r => r.data)