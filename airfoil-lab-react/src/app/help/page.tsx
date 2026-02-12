'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Header } from '@/components/layout/Header';

export default function HelpPage() {
    return (
        <div className="min-h-screen bg-[hsl(220,20%,10%)]">
            <Header />

            <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="card-panel p-8 space-y-8">
                    <div>
                        <h2 className="text-2xl font-bold text-white mb-2">❓ Help Center</h2>
                        <p className="text-sm text-slate-400">Everything you need to know about AI-Enhanced Airfoil Design Lab</p>
                    </div>

                    <div className="space-y-6 text-slate-300 text-sm leading-relaxed">
                        <div>
                            <h3 className="text-lg font-semibold text-white mb-3">Welcome to AI-Enhanced Airfoil Design Lab</h3>
                            <p className="text-slate-300 mb-3">This app combines:</p>
                            <ul className="space-y-1 text-slate-300 list-disc ml-5">
                                <li><strong className="text-white">Airfoil geometry generation</strong> (NACA 4-digit + thickness position)</li>
                                <li><strong className="text-white">XFOIL performance analysis</strong> (polar curves + Cp distribution)</li>
                                <li><strong className="text-white">AI tutoring chat</strong> (multi-module intelligent assistant)</li>
                                <li><strong className="text-white">Downloads + personal history</strong></li>
                            </ul>
                        </div>

                        <div className="border-t border-[hsl(220,15%,22%)] pt-6">
                            <h3 className="text-lg font-semibold text-white mb-3">1) Quick Start</h3>
                            <ol className="space-y-1.5 text-slate-300 list-decimal ml-5">
                                <li>Go to the main <strong className="text-white">Geometry</strong> page</li>
                                <li>Adjust <strong className="text-white">Camber / Thickness / Max thickness position / α</strong></li>
                                <li>Click <strong className="text-white">Run Simulation</strong> to execute XFOIL analysis</li>
                                <li>Review KPI cards (<strong className="text-white">CL, CD, L/D, max L/D</strong>)</li>
                                <li>Use polar charts to analyze performance</li>
                                <li>Download datasets (CSV) for reports</li>
                            </ol>
                        </div>

                        <div className="border-t border-white/5 pt-6">
                            <h3 className="text-lg font-semibold text-white mb-3">2) Key Parameters</h3>
                            <div className="overflow-x-auto rounded-lg border border-[hsl(220,15%,22%)]">
                                <table className="min-w-full text-sm">
                                    <thead className="bg-[hsl(220,15%,18%)]">
                                        <tr>
                                            <th className="px-4 py-2 text-left text-slate-200 font-medium">Parameter</th>
                                            <th className="px-4 py-2 text-left text-slate-200 font-medium">Description</th>
                                            <th className="px-4 py-2 text-left text-slate-200 font-medium">Typical Range</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-[hsl(220,15%,20%)] text-slate-300">
                                        <tr>
                                            <td className="px-4 py-2"><strong className="text-white">Camber (%)</strong></td>
                                            <td className="px-4 py-2">Maximum camber as percentage of chord</td>
                                            <td className="px-4 py-2 font-mono">0-10%</td>
                                        </tr>
                                        <tr>
                                            <td className="px-4 py-2"><strong className="text-slate-300">Max Camber Pos (%)</strong></td>
                                            <td className="px-4 py-2">Location of maximum camber</td>
                                            <td className="px-4 py-2 font-mono">20-60%</td>
                                        </tr>
                                        <tr>
                                            <td className="px-4 py-2"><strong className="text-slate-300">Thickness (%)</strong></td>
                                            <td className="px-4 py-2">Maximum thickness as percentage of chord</td>
                                            <td className="px-4 py-2 font-mono">5-20%</td>
                                        </tr>
                                        <tr>
                                            <td className="px-4 py-2"><strong className="text-slate-300">Max Thickness Pos (%)</strong></td>
                                            <td className="px-4 py-2">Location of maximum thickness</td>
                                            <td className="px-4 py-2 font-mono">20-50%</td>
                                        </tr>
                                        <tr>
                                            <td className="px-4 py-2"><strong className="text-slate-300">α (Angle of Attack)</strong></td>
                                            <td className="px-4 py-2">Angle between chord and freestream</td>
                                            <td className="px-4 py-2 font-mono">-10° to 15°</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div className="border-t border-white/5 pt-6">
                            <h3 className="text-lg font-semibold text-white mb-3">3) AI Chat Modules</h3>
                            <ul className="space-y-1 text-slate-300 list-disc ml-5">
                                <li><strong className="text-blue-400">Concept Learning</strong> - Learn aerodynamics fundamentals</li>
                                <li><strong className="text-amber-400">Model Iteration</strong> - Help with experiment design and parameter tuning</li>
                                <li><strong className="text-purple-400">Strategy Review</strong> - Get feedback on your design approach</li>
                            </ul>
                        </div>

                        <div className="border-t border-white/5 pt-6">
                            <h3 className="text-lg font-semibold text-white mb-3">4) Troubleshooting</h3>
                            <p className="text-slate-200 mb-2"><strong>XFOIL did not converge / Empty results:</strong></p>
                            <ul className="space-y-1 text-slate-300 list-disc ml-5 mb-4">
                                <li>Reduce α range or use smaller step size</li>
                                <li>Adjust Ncrit value (7-9 is typical)</li>
                                <li>Ensure xfoil.exe is in the backend directory</li>
                            </ul>

                            <p className="text-slate-200 mb-2"><strong>Backend connection failed:</strong></p>
                            <ul className="space-y-1 text-slate-300 list-disc ml-5">
                                <li>Make sure the Python backend is running on port 8000</li>
                                <li>Check NEXT_PUBLIC_API_URL in .env.local</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
