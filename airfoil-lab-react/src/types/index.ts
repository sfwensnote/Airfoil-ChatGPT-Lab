// TypeScript 类型定义

// 翼型几何参数
export interface GeometryParams {
  camber: number;       // 弯度 (0-10%)
  maxCamberPos: number; // 最大弯度位置 (0-100%)
  thickness: number;    // 厚度 (5-20%)
  maxThicknessPos: number; // 最大厚度位置 (0-100%)
}

// 环境参数
export interface EnvironmentParams {
  rho: number;      // 空气密度 (kg/m³)
  velocity: number; // 速度 (m/s)
  chord: number;    // 弦长 (m)
  mu: number;       // 动力粘度
  mach: number;     // 马赫数
  ncrit: number;    // Ncrit
  alpha: number;    // 攻角 (°)
  alphaRange: [number, number]; // 扫描范围
  alphaStep: number; // 步长
}

// 仿真结果
export interface SimulationResult {
  polar: PolarPoint[];
  kpi: KPIData;
  geometry: AirfoilGeometry;
  cpData?: CpDataPoint[];
}

export interface PolarPoint {
  alpha: number;
  CL: number;
  CD: number;
  CM: number;
}

export interface KPIData {
  cl: number;
  cd: number;
  ld: number;
  alphaOpt: number;
  ldMax: number;
}

export interface AirfoilGeometry {
  x: number[];
  y: number[];
  nacaCode: string;
}

export interface CpDataPoint {
  segment: 'upper' | 'lower';
  x: number;
  y?: number;
  cp: number;
}

// 对话消息
export interface ChatMessage {
  id: string;
  role: 'user' | 'ai' | 'system';
  content: string;
  module?: 'Concept Learning' | 'Model Iteration' | 'Strategy Review';
  timestamp: string;
}

// AI 模块配置
export interface AIModule {
  name: string;
  description: string;
  color: string;
}

// 历史记录
export interface AirfoilHistory {
  id: number;
  userId: string;
  nacaCode: string;
  camber: number;
  thickness: number;
  maxCamberPos: number;
  maxThicknessPos: number;
  alpha: number;
  rho: number;
  velocity: number;
  chord: number;
  mu: number;
  re: number;
  ncrit: number;
  mach: number;
  cl: number;
  cd: number;
  ld: number;
  alphaOpt: number;
  ldMax: number;
  timestamp: string;
}

// API 响应
export interface ApiResponse<T> {
  status: 'success' | 'error';
  data?: T;
  message?: string;
}
