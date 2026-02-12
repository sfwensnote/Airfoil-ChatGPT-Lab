'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Loader2, Play, Download, MessageSquare,
  PanelLeftClose, PanelLeftOpen, Eye, EyeOff,
  Settings2, ChevronDown, ChevronUp
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { AirfoilPreview } from '@/components/airfoil/AirfoilPreview';
import { PolarCharts } from '@/components/airfoil/PolarCharts';
import { KPICards } from '@/components/airfoil/KPICards';
import { CpChart } from '@/components/airfoil/CpChart';
import { ParameterSlider } from '@/components/controls/ParameterSlider';
import { ChatDrawer } from '@/components/chat/ChatDrawer';
import { Header } from '@/components/layout/Header';
import { useSimulationStore, useUserStore } from '@/stores';
import { genNaca4, nacaCodeFromMpt, estimateRe } from '@/lib/geometry';

export default function HomePage() {
  const [isSimulating, setIsSimulating] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showPressure, setShowPressure] = useState(false);
  const [resultSource, setResultSource] = useState<'realtime' | 'xfoil'>('realtime');

  const { geometry, environment, result, setGeometry, setEnvironment, setResult } = useSimulationStore();
  const { currentUser, isAuthenticated } = useUserStore();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, router]);

  // Keyboard shortcut: Ctrl+J to toggle chat
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'j') {
        e.preventDefault();
        setChatOpen(prev => !prev);
      }
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, []);

  // Real-time Simulation Effect
  useEffect(() => {
    runRealTimeSimulation();
    setResultSource('realtime');
  }, [geometry, environment]);

  const userId = currentUser ? currentUser.username : 'guest';

  const re = estimateRe(
    environment.rho,
    environment.velocity,
    environment.chord,
    environment.mu
  );

  const nacaCode = nacaCodeFromMpt(
    geometry.camber / 100,
    geometry.maxCamberPos / 100,
    geometry.thickness / 100
  );

  const handleRunSimulation = async () => {
    setIsSimulating(true);
    // Auto-enable pressure view on run if not already
    if (!showPressure) setShowPressure(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camber: geometry.camber / 100,
          thickness: geometry.thickness / 100,
          max_camber_pos: geometry.maxCamberPos / 100,
          max_thickness_pos: geometry.maxThicknessPos / 100,
          alpha: environment.alpha,
          rho: environment.rho,
          velocity: environment.velocity,
          chord: environment.chord,
          mu: environment.mu,
          ncrit: environment.ncrit,
          mach: environment.mach,
          alpha_start: environment.alphaRange[0],
          alpha_end: environment.alphaRange[1],
          alpha_step: environment.alphaStep,
          user_id: userId,
        }),
      });
      const result = await response.json();
      if (result.status === 'success' && result.data) {
        setResult(result.data);
        setResultSource('xfoil');
      } else {
        console.error('Simulation failed:', result.warning);
        // Don't fallback to realtime here to strictly separate sources, or maybe notify user
        // runRealTimeSimulation(); 
      }
    } catch (error) {
      console.error('API call failed, using fallback:', error);
      // runRealTimeSimulation();
    }
    setIsSimulating(false);
  };

  const runRealTimeSimulation = () => {
    const coords = genNaca4(
      geometry.camber / 100,
      geometry.maxCamberPos / 100,
      geometry.thickness / 100,
      geometry.maxThicknessPos / 100
    );
    // ... logic same as fallbackSimulation ...
    const polarData = [];
    for (let alpha = environment.alphaRange[0]; alpha <= environment.alphaRange[1]; alpha += environment.alphaStep) {
      const cl = 0.11 * alpha + 0.2 * (geometry.camber / 100) * 10;
      const cd = 0.008 + 0.0001 * alpha * alpha + 0.001 * (geometry.thickness / 100);
      polarData.push({ alpha, CL: cl, CD: cd, CM: -0.05 - 0.002 * alpha });
    }
    let maxLD = 0, alphaOpt = 0;
    polarData.forEach((p) => {
      const ld = p.CL / p.CD;
      if (ld > maxLD) { maxLD = ld; alphaOpt = p.alpha; }
    });
    const currentPoint = polarData.find((p) => Math.abs(p.alpha - environment.alpha) < 0.1) || polarData[0];

    // Generate fallback Cp data
    const cpData = generateFallbackCp(
      geometry.camber / 100,
      geometry.thickness / 100,
      geometry.maxCamberPos / 100,
      environment.alpha
    );

    setResult({
      polar: polarData,
      kpi: { cl: currentPoint.CL, cd: currentPoint.CD, ld: currentPoint.CL / currentPoint.CD, alphaOpt, ldMax: maxLD },
      geometry: { x: coords.x, y: coords.y, nacaCode: nacaCodeFromMpt(geometry.camber / 100, geometry.maxCamberPos / 100, geometry.thickness / 100) },
      cpData,
    });
  };

  // Generate approximate Cp distribution using thin-airfoil theory
  const generateFallbackCp = (camber: number, thickness: number, maxCamberPos: number, alpha: number) => {
    const n = 40;
    const upper: { segment: 'upper'; x: number; cp: number }[] = [];
    const lower: { segment: 'lower'; x: number; cp: number }[] = [];
    const alphaRad = alpha * Math.PI / 180;

    for (let i = 0; i <= n; i++) {
      const theta = (Math.PI * i) / n;
      const x = (1 - Math.cos(theta)) / 2;

      // Approximate velocity ratio from thin-airfoil theory
      const camberEffect = camber > 0 ? 2 * camber * (1 - 2 * x) / Math.max(maxCamberPos, 0.01) : 0;
      const thicknessEffect = thickness * (1 - 2 * x * x);

      // Upper surface: accelerated flow
      const vUpper = 1 + alphaRad * (1 - x) / Math.max(x, 0.01) * 0.05 + thicknessEffect + camberEffect * 0.5;
      const cpUpper = 1 - vUpper * vUpper;
      upper.push({ segment: 'upper', x, cp: Math.max(-6, Math.min(1, cpUpper)) });

      // Lower surface: decelerated flow
      const vLower = 1 - alphaRad * (1 - x) / Math.max(x, 0.01) * 0.03 - thicknessEffect * 0.5 - camberEffect * 0.3;
      const cpLower = 1 - vLower * vLower;
      lower.push({ segment: 'lower', x, cp: Math.max(-6, Math.min(1, cpLower)) });
    }

    return [...upper, ...lower];
  };

  return (
    <div className="h-screen flex flex-col bg-[hsl(220,20%,10%)] overflow-hidden">
      <Header />

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside
          className={`
            flex-shrink-0 bg-[hsl(220,20%,12%)] border-r border-[hsl(220,15%,20%)] 
            transition-all duration-300 ease-in-out flex flex-col
            ${sidebarOpen ? 'w-80' : 'w-0 opacity-0 overflow-hidden'}
          `}
        >
          <div className="p-4 border-b border-[hsl(220,15%,20%)] flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Settings2 className="w-4 h-4 text-emerald-400" />
              Design Parameters
            </h2>
          </div>

          <div className="flex-1 overflow-y-auto p-5 space-y-6">
            <ParameterSlider
              label="Camber"
              value={geometry.camber}
              min={0} max={10} step={0.1} unit="%"
              onChange={(v) => setGeometry({ camber: v })}
            />
            <ParameterSlider
              label="Thickness"
              value={geometry.thickness}
              min={5} max={20} step={0.1} unit="%"
              onChange={(v) => setGeometry({ thickness: v })}
            />
            <ParameterSlider
              label="Max Camber Position"
              value={geometry.maxCamberPos}
              min={0} max={100} step={1} unit="%"
              onChange={(v) => setGeometry({ maxCamberPos: v })}
              formatValue={(v) => v.toFixed(0)}
            />
            <ParameterSlider
              label="Max Thickness Position"
              value={geometry.maxThicknessPos}
              min={0} max={100} step={1} unit="%"
              onChange={(v) => setGeometry({ maxThicknessPos: v })}
              formatValue={(v) => v.toFixed(0)}
            />

            <Separator className="bg-[hsl(220,15%,20%)]" />

            <ParameterSlider
              label="Angle of Attack (α)"
              value={environment.alpha}
              min={-10} max={15} step={0.5} unit="°"
              onChange={(v) => setEnvironment({ alpha: v })}
            />

            <Button
              onClick={handleRunSimulation}
              disabled={isSimulating}
              className="w-full mt-4 h-11 text-base font-semibold bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 border-0 glow-button transition-all duration-300 cursor-pointer"
            >
              {isSimulating ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Running (XFOIL)...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 mr-2 animate-pulse" />
                  Run Accurate Simulation
                </>
              )}
            </Button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto bg-[hsl(220,20%,10%)] relative">

          {/* Sidebar Toggle (Floating) */}
          <div className="absolute top-4 left-4 z-10">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="bg-[hsl(220,20%,14%)]/80 backdrop-blur border border-[hsl(220,15%,25%)] text-slate-400 hover:text-white hover:bg-[hsl(220,15%,20%)] w-8 h-8 p-0 rounded-full shadow-lg"
              title={sidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
            >
              {sidebarOpen ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeftOpen className="w-4 h-4" />}
            </Button>
          </div>

          <div className="max-w-6xl mx-auto px-6 py-4 space-y-4">

            {/* Airfoil Preview - Hero */}
            <div className="card-panel overflow-hidden glow-blue relative">
              <div className="h-[280px] w-full">
                <AirfoilPreview
                  camber={geometry.camber}
                  maxCamberPos={geometry.maxCamberPos}
                  thickness={geometry.thickness}
                  maxThicknessPos={geometry.maxThicknessPos}
                  alpha={environment.alpha}
                  cpData={result?.cpData}
                  showPressure={showPressure}
                  className="w-full h-full"
                />
              </div>

              {/* Pressure Toggle Overlay */}
              <div className="absolute bottom-3 right-3 flex items-center gap-2 bg-[hsl(220,20%,10%)]/80 backdrop-blur px-3 py-1.5 rounded-full border border-[hsl(220,15%,25%)]">
                <Label htmlFor="pressure-toggle" className="text-xs text-slate-300 font-medium cursor-pointer">Pressure Vectors</Label>
                <Switch
                  id="pressure-toggle"
                  checked={showPressure}
                  onCheckedChange={setShowPressure}
                  className="scale-75"
                />
              </div>
            </div>

            {/* Status Bar */}
            <div className="flex items-center gap-4 card-panel px-5 py-3 justify-between">
              <div className="flex flex-wrap items-center gap-4 text-sm text-slate-300">
                <span>NACA <strong className="text-white font-mono">{nacaCode}</strong></span>
                <span className="text-slate-500">•</span>
                <span>Re ≈ <strong className="text-white font-mono">{re.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong></span>
                <span className="text-slate-500">•</span>
                <span>M = <strong className="text-white font-mono">{environment.mach.toFixed(2)}</strong></span>
                <span className="text-slate-500">•</span>
                <span>α = <strong className="text-white font-mono">{environment.alpha.toFixed(1)}°</strong></span>
              </div>

              {/* Result Source Badge */}
              <div className={`px-2 py-0.5 rounded textxs font-semibold uppercase tracking-wider ${resultSource === 'xfoil' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'}`}>
                {resultSource === 'xfoil' ? 'XFOIL Analysis' : 'Real-time Estimate'}
              </div>
            </div>

            {/* KPI Cards */}
            {result && <KPICards data={result.kpi} />}

            {/* Charts Section */}
            {result && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Performance Charts */}
                <div className="card-panel p-5">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
                    <span>📊</span>
                    Performance Analysis
                  </h3>
                  <PolarCharts
                    data={result.polar}
                    currentAlpha={environment.alpha}
                    optimalAlpha={result.kpi.alphaOpt}
                  />
                </div>

                {/* Cp Distribution Chart */}
                <div className="card-panel p-5">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
                    <span>📈</span>
                    Pressure Distribution (Cp)
                  </h3>
                  <CpChart
                    data={result.cpData || []}
                    nacaCode={result.geometry.nacaCode}
                    alpha={environment.alpha}
                  />
                </div>
              </div>
            )}

            {/* Download Buttons */}
            {/* Data Export Section */}
            {result && (
              <div className="card-panel p-5">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-4">
                  <span>💾</span>
                  Data Export
                </h3>
                <div className="flex flex-wrap gap-4">
                  <Button
                    className="flex-1 h-12 bg-[hsl(220,20%,16%)] hover:bg-[hsl(220,20%,20%)] border border-[hsl(220,15%,25%)] text-slate-200 hover:text-white transition-all duration-200 group relative overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <Download className="w-5 h-5 mr-3 text-blue-400 group-hover:scale-110 transition-transform" />
                    <div className="flex flex-col items-start gap-0.5 z-10">
                      <span className="text-sm font-medium">Geometry Data</span>
                      <span className="text-[10px] text-slate-400">CSV Coordinate Points</span>
                    </div>
                  </Button>

                  <Button
                    className="flex-1 h-12 bg-[hsl(220,20%,16%)] hover:bg-[hsl(220,20%,20%)] border border-[hsl(220,15%,25%)] text-slate-200 hover:text-white transition-all duration-200 group relative overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <Download className="w-5 h-5 mr-3 text-emerald-400 group-hover:scale-110 transition-transform" />
                    <div className="flex flex-col items-start gap-0.5 z-10">
                      <span className="text-sm font-medium">Polar Analysis</span>
                      <span className="text-[10px] text-slate-400">CSV Performance Data</span>
                    </div>
                  </Button>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Floating AI Chat Button (FAB) */}
      <button
        onClick={() => setChatOpen(true)}
        className="fixed bottom-6 right-6 z-40 w-14 h-14 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 text-white shadow-lg flex items-center justify-center hover:scale-105 active:scale-95 transition-transform duration-200 fab-pulse cursor-pointer"
        title="AI Tutor Chat (Ctrl+J)"
      >
        <MessageSquare className="w-6 h-6" />
      </button>

      {/* Chat Drawer */}
      <ChatDrawer isOpen={chatOpen} onClose={() => setChatOpen(false)} />
    </div>
  );
}
