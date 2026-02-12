import { create } from 'zustand';
import {
    GeometryParams,
    EnvironmentParams,
    SimulationResult,
    ChatMessage,
    AirfoilHistory
} from '@/types';

// ============= 仿真状态 =============
interface SimulationState {
    // 几何参数
    geometry: GeometryParams;
    // 环境参数
    environment: EnvironmentParams;
    // 仿真结果
    result: SimulationResult | null;
    // 加载状态
    isLoading: boolean;
    // 错误信息
    error: string | null;

    // Actions
    setGeometry: (params: Partial<GeometryParams>) => void;
    setEnvironment: (params: Partial<EnvironmentParams>) => void;
    setResult: (result: SimulationResult | null) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    reset: () => void;
}

const defaultGeometry: GeometryParams = {
    camber: 2.0,
    maxCamberPos: 40.0,
    thickness: 12.0,
    maxThicknessPos: 30.0,
};

const defaultEnvironment: EnvironmentParams = {
    rho: 1.225,
    velocity: 10.0,
    chord: 1.0,
    mu: 1.8e-5,
    mach: 0.0,
    ncrit: 7.0,
    alpha: 5.0,
    alphaRange: [-2.0, 10.0],
    alphaStep: 0.5,
};

export const useSimulationStore = create<SimulationState>((set) => ({
    geometry: defaultGeometry,
    environment: defaultEnvironment,
    result: null,
    isLoading: false,
    error: null,

    setGeometry: (params) => set((state) => ({
        geometry: { ...state.geometry, ...params },
    })),

    setEnvironment: (params) => set((state) => ({
        environment: { ...state.environment, ...params },
    })),

    setResult: (result) => set({ result }),

    setLoading: (isLoading) => set({ isLoading }),

    setError: (error) => set({ error }),

    reset: () => set({
        geometry: defaultGeometry,
        environment: defaultEnvironment,
        result: null,
        isLoading: false,
        error: null,
    }),
}));

// ============= 对话状态 =============
interface ChatState {
    messages: ChatMessage[];
    currentModule: 'Concept Learning' | 'Model Iteration' | 'Strategy Review';
    isStreaming: boolean;
    userId: string;

    // Actions
    addMessage: (message: ChatMessage) => void;
    updateMessage: (id: string, updates: Partial<ChatMessage>) => void;
    setModule: (module: ChatState['currentModule']) => void;
    setStreaming: (streaming: boolean) => void;
    setUserId: (userId: string) => void;
    clearMessages: () => void;
    loadHistory: (messages: ChatMessage[]) => void;
}

export const useChatStore = create<ChatState>((set) => ({
    messages: [],
    currentModule: 'Concept Learning',
    isStreaming: false,
    userId: 'guest',

    addMessage: (message) => set((state) => ({
        messages: [...state.messages, message],
    })),

    updateMessage: (id, updates) => set((state) => ({
        messages: state.messages.map((m) =>
            m.id === id ? { ...m, ...updates } : m
        ),
    })),

    setModule: (currentModule) => set({ currentModule }),

    setStreaming: (isStreaming) => set({ isStreaming }),

    setUserId: (userId) => set({ userId }),

    clearMessages: () => set({ messages: [] }),

    loadHistory: (messages) => set({ messages }),
}));

// ============= 用户状态 =============
// ============= 用户状态 =============
interface User {
    id: number;
    username: string;
    role: string;
}

interface UserState {
    currentUser: User | null;
    isAuthenticated: boolean;
    history: AirfoilHistory[];

    // Actions
    login: (user: User) => void;
    logout: () => void;
    setHistory: (history: AirfoilHistory[]) => void;
}

export const useUserStore = create<UserState>((set) => ({
    currentUser: null,
    isAuthenticated: false,
    history: [],

    login: (user) => {
        localStorage.setItem('auth_user', JSON.stringify(user));
        set({ currentUser: user, isAuthenticated: true });
    },
    logout: () => {
        localStorage.removeItem('auth_user');
        set({ currentUser: null, isAuthenticated: false });
    },
    setHistory: (history) => set({ history }),
}));

// ============= UI状态 =============
interface UIState {
    sidebarOpen: boolean;
    activeTab: 'geometry' | 'history' | 'help' | 'admin';

    // Actions
    setSidebarOpen: (open: boolean) => void;
    setActiveTab: (tab: UIState['activeTab']) => void;
}

export const useUIStore = create<UIState>((set) => ({
    sidebarOpen: true,
    activeTab: 'geometry',

    setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
    setActiveTab: (activeTab) => set({ activeTab }),
}));
