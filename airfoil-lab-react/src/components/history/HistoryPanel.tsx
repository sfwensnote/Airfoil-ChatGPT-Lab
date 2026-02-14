
'use client';

import { useState, useEffect } from 'react';
import { useUserStore, useSimulationStore } from '@/stores';
import { getAirfoilHistory } from '@/lib/api';
import { AirfoilHistory } from '@/types';
import { Button } from '@/components/ui/button';
import { Loader2, RefreshCw, RotateCcw, ArrowLeft, TrendingUp } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import { Header } from '@/components/layout/Header';
import { HistoryChart } from '@/components/history/HistoryChart';

export function HistoryPanel() {
    const router = useRouter();
    const { currentUser } = useUserStore();
    const userId = currentUser ? currentUser.username : 'guest';
    const { setGeometry, setEnvironment } = useSimulationStore();
    const [history, setHistory] = useState<AirfoilHistory[]>([]);
    const [loading, setLoading] = useState(false);

    const loadData = async () => {
        if (!userId || userId === 'guest') return;
        setLoading(true);
        try {
            const data = await getAirfoilHistory(userId);
            // Sanitize data: ensure maxThicknessPos is valid number
            const sanitized = data.map((item: AirfoilHistory) => ({
                ...item,
                maxThicknessPos: item.maxThicknessPos || 0.3,
                timestamp: item.timestamp || new Date().toISOString()
            }));
            sanitized.sort((a: AirfoilHistory, b: AirfoilHistory) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
            setHistory(sanitized);
        } catch (error) {
            console.error('Failed to load history:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
    }, [userId]);

    const handleRestore = (item: AirfoilHistory) => {
        setGeometry({
            camber: item.camber * 100,
            thickness: item.thickness * 100,
            maxCamberPos: item.maxCamberPos * 100,
            maxThicknessPos: (item.maxThicknessPos || 0.3) * 100,
        });

        setEnvironment({
            alpha: item.alpha,
            mach: item.mach,
            ncrit: item.ncrit,
            velocity: item.velocity,
            rho: item.rho,
            mu: item.mu,
            chord: item.chord,
            alphaRange: [-10, 20],
            alphaStep: 1,
        });

        toast.success(`Restored parameters for NACA ${item.nacaCode}`);
        router.push('/');
    };

    if (!userId || userId === 'guest') {
        return (
            <div className="min-h-screen bg-[hsl(220,20%,10%)]">
                <Header />
                <div className="flex flex-col items-center justify-center p-12 text-center h-[60vh]">
                    <h2 className="text-xl font-semibold text-slate-200 mb-2">Please Login</h2>
                    <p className="text-slate-400 mb-4">Enter your User ID in the header to view history.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
            <Header />
            <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link href="/">
                            <Button variant="ghost" size="sm" className="gap-2 text-slate-300 hover:text-white hover:bg-[hsl(220,15%,20%)] cursor-pointer">
                                <ArrowLeft className="w-4 h-4" />
                                Back to Lab
                            </Button>
                        </Link>
                        <h1 className="text-2xl font-bold text-slate-100">Simulation History</h1>
                    </div>
                    <Button onClick={loadData} disabled={loading} variant="outline" className="gap-2 border-[hsl(220,15%,25%)] text-slate-200 hover:bg-[hsl(220,15%,20%)] cursor-pointer">
                        <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                        Refresh
                    </Button>
                </div>

                <div className="card-panel p-6">
                    <div className="mb-4">
                        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                            <TrendingUp className="w-5 h-5 text-emerald-400" />
                            Design Iteration Trends
                        </h2>
                        <p className="text-sm text-slate-400">
                            Track performance metrics across your design iterations. Scroll horizontally to see all data points.
                        </p>
                    </div>
                    <HistoryChart data={history} />
                </div>

                <div className="card-panel p-6">
                    <div className="mb-4">
                        <h2 className="text-lg font-semibold text-white">Design Iterations</h2>
                        <p className="text-sm text-slate-400">
                            Review your past airfoil designs and performance metrics. Click &ldquo;Restore&rdquo; to load parameters back into the lab.
                        </p>
                    </div>

                    <div className="overflow-auto rounded-lg border border-[hsl(220,15%,22%)] max-h-[480px]">
                        <table className="min-w-full text-sm">
                            <thead className="bg-[hsl(220,15%,18%)] sticky top-0 z-10">
                                <tr>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium">NACA</th>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium">Camber</th>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium">Thick</th>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium">L/D Max</th>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium">CL Max</th>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium hidden md:table-cell">Re</th>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium hidden md:table-cell">Time</th>
                                    <th className="px-4 py-3 text-right text-slate-300 font-medium">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-[hsl(220,15%,20%)]">
                                {loading && history.length === 0 ? (
                                    <tr>
                                        <td colSpan={8} className="text-center py-8">
                                            <Loader2 className="w-6 h-6 animate-spin mx-auto text-slate-500" />
                                        </td>
                                    </tr>
                                ) : history.length === 0 ? (
                                    <tr>
                                        <td colSpan={8} className="text-center py-8 text-slate-500">
                                            No simulation history found for user &ldquo;{userId}&rdquo;.
                                        </td>
                                    </tr>
                                ) : (
                                    history.map((item) => (
                                        <tr key={item.id} className="hover:bg-[hsl(220,15%,18%)] transition-colors">
                                            <td className="px-4 py-3 font-medium text-white font-mono">{item.nacaCode}</td>
                                            <td className="px-4 py-3 text-slate-300">{(item.camber * 100).toFixed(0)}%</td>
                                            <td className="px-4 py-3 text-slate-300">{(item.thickness * 100).toFixed(0)}%</td>
                                            <td className="px-4 py-3 font-bold text-emerald-400 font-mono">
                                                {item.ldMax?.toFixed(1) || '-'}
                                            </td>
                                            <td className="px-4 py-3 text-slate-300">{item.cl?.toFixed(3) || '-'}</td>
                                            <td className="px-4 py-3 hidden md:table-cell text-slate-300 font-mono">
                                                {(item.re / 1e6).toFixed(2)}M
                                            </td>
                                            <td className="px-4 py-3 hidden md:table-cell text-xs text-slate-500">
                                                {new Date(item.timestamp).toLocaleString()}
                                            </td>
                                            <td className="px-4 py-3 text-right">
                                                <Button
                                                    size="sm"
                                                    variant="ghost"
                                                    onClick={() => handleRestore(item)}
                                                    className="h-8 w-8 p-0 text-slate-300 hover:text-white hover:bg-[hsl(220,15%,20%)] cursor-pointer"
                                                    title="Restore Parameters"
                                                >
                                                    <RotateCcw className="w-4 h-4" />
                                                </Button>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default HistoryPanel;
