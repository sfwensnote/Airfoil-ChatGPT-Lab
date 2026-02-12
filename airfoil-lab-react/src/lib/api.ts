import axios from 'axios';
import {
    SimulationResult,
    GeometryParams,
    EnvironmentParams,
    AirfoilHistory,
    ChatMessage,
    ApiResponse
} from '@/types';

// 后端API基础URL - 可通过环境变量配置
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 90000, // XFOIL仿真可能需要较长时间
    headers: {
        'Content-Type': 'application/json',
    },
});

// ============= 仿真 API =============

export interface SimulateParams {
    geometry: GeometryParams;
    environment: EnvironmentParams;
    userId: string;
}

/**
 * 执行XFOIL仿真
 */
export async function runSimulation(params: SimulateParams): Promise<SimulationResult> {
    const { geometry, environment, userId } = params;

    const response = await api.post<ApiResponse<SimulationResult>>('/simulate', {
        user_id: userId,
        naca_code: '', // 后端会根据参数生成
        camber: geometry.camber / 100,
        thickness: geometry.thickness / 100,
        max_camber_pos: geometry.maxCamberPos / 100,
        max_thickness_pos: geometry.maxThicknessPos / 100,
        alpha: environment.alpha,
        rho: environment.rho,
        velocity: environment.velocity,
        chord: environment.chord,
        mu: environment.mu,
        re: 0, // 后端计算
        ncrit: environment.ncrit,
        mach: environment.mach,
        alpha_start: environment.alphaRange[0],
        alpha_end: environment.alphaRange[1],
        alpha_step: environment.alphaStep,
    });

    if (response.data.status === 'error') {
        throw new Error(response.data.message || 'Simulation failed');
    }

    return response.data.data!;
}

/**
 * 获取Cp分布数据
 */
export async function getCpDistribution(
    geometry: GeometryParams,
    environment: EnvironmentParams,
    alpha: number
): Promise<{ upper: { x: number; cp: number }[]; lower: { x: number; cp: number }[] }> {
    const response = await api.post('/cp', {
        camber: geometry.camber / 100,
        thickness: geometry.thickness / 100,
        max_camber_pos: geometry.maxCamberPos / 100,
        max_thickness_pos: geometry.maxThicknessPos / 100,
        alpha,
        rho: environment.rho,
        velocity: environment.velocity,
        chord: environment.chord,
        mu: environment.mu,
        ncrit: environment.ncrit,
        mach: environment.mach,
    });

    return response.data.data;
}

// ============= 历史 API =============

/**
 * 获取用户翼型历史
 */
export async function getAirfoilHistory(userId: string): Promise<AirfoilHistory[]> {
    const response = await api.get<any[]>(`/export_airfoils/${userId}`);
    return response.data.map(item => ({
        id: item.id,
        userId: item.user_id,
        nacaCode: item.naca_code,
        camber: item.camber,
        thickness: item.thickness,
        maxCamberPos: item.max_camber_pos,
        maxThicknessPos: item.max_thickness_pos,
        alpha: item.alpha,
        rho: item.rho,
        velocity: item.velocity,
        chord: item.chord,
        mu: item.mu,
        re: item.re,
        ncrit: item.ncrit,
        mach: item.mach,
        cl: item.cl,
        cd: item.cd,
        ld: item.ld,
        alphaOpt: item.alpha_opt,
        ldMax: item.ld_max,
        timestamp: item.timestamp,
    }));
}

/**
 * 保存翼型数据
 */
export async function saveAirfoil(data: Omit<AirfoilHistory, 'id' | 'timestamp'>): Promise<void> {
    await api.post('/save_airfoil/', {
        user_id: data.userId,
        naca_code: data.nacaCode,
        camber: data.camber,
        thickness: data.thickness,
        max_camber_pos: data.maxCamberPos,
        max_thickness_pos: data.maxThicknessPos,
        alpha: data.alpha,
        rho: data.rho,
        velocity: data.velocity,
        chord: data.chord,
        mu: data.mu,
        re: data.re,
        ncrit: data.ncrit,
        mach: data.mach,
        cl: data.cl,
        cd: data.cd,
        ld: data.ld,
        alpha_opt: data.alphaOpt,
        ld_max: data.ldMax,
    });
}

// ============= 对话 API =============

/**
 * 获取对话历史
 */
export async function getConversationHistory(userId: string): Promise<ChatMessage[]> {
    const response = await api.get(`/export_conversations/${userId}`);
    const messages: ChatMessage[] = [];

    response.data.forEach((item: Record<string, unknown>) => {
        const timestamp = item.timestamp as string;
        const moduleId = item.role as ChatMessage['module'];

        // Add User Message
        if (item.student_question) {
            messages.push({
                id: `q-${item.id}`,
                role: 'user',
                content: item.student_question as string,
                timestamp: timestamp,
                module: moduleId, // Associate user question with the handling agent
            });
        }

        // Add AI Message
        if (item.ai_response) {
            messages.push({
                id: `a-${item.id}`,
                role: 'ai',
                content: item.ai_response as string,
                module: moduleId,
                timestamp: timestamp,
            });
        }
    });

    return messages;
}

/**
 * 保存对话
 */
export async function saveConversation(
    userId: string,
    role: string,
    question: string,
    response: string
): Promise<void> {
    await api.post('/save_conversation/', {
        user_id: userId,
        role,
        student_question: question,
        ai_response: response,
    });
}

/**
 * 删除对话历史 (用户级)
 */
export async function deleteConversationHistory(userId: string): Promise<void> {
    await api.delete(`/delete_history/${userId}`);
}

// ============= 管理员 API =============

/**
 * 导出所有对话 (管理员)
 */
export async function exportAllConversations(): Promise<Blob> {
    const response = await api.get('/admin/export_all_conversations', {
        responseType: 'blob',
    });
    return response.data;
}

/**
 * 导出所有翼型数据 (管理员)
 */
export async function exportAllAirfoils(): Promise<Blob> {
    const response = await api.get('/admin/export_all_airfoils', {
        responseType: 'blob',
    });
    return response.data;
}

// ============= Multi-Agent Chat API =============

export interface AgentChatRequest {
    message: string;
    userId: string;
    preferredAgent?: string;  // 用户选择的模块 (覆盖自动路由)
    // 可选值: "auto", "Concept Learning", "Model Iteration", "Strategy Review"
    context?: {
        geometry?: {
            camber: number;
            thickness: number;
            maxCamberPos: number;
            maxThicknessPos: number;
            nacaCode?: string;
        };
        environment?: {
            alpha: number;
            re: number;
            velocity: number;
            rho: number;
            mu: number;
        };
        kpi?: {
            cl: number;
            cd: number;
            ld: number;
            alphaOpt: number;
            ldMax: number;
        };
        history?: Array<{
            naca_code: string;
            cl: number;
            cd: number;
            ld: number;
        }>;
    };
}

export interface AgentChatResponse {
    status: string;
    agent: 'concept_mentor' | 'iteration_engineer' | 'strategy_analyst';
    agent_display_name: string;
    response: string;
    message?: string;
}

/**
 * Chat with the multi-agent system
 * Supports hybrid routing:
 * - If preferredAgent is "auto" or not set, uses automatic intent-based routing
 * - If preferredAgent is set to a module name, uses that agent directly
 */
export async function chatWithAgent(request: AgentChatRequest): Promise<AgentChatResponse> {
    const response = await api.post<AgentChatResponse>('/agent/chat', {
        message: request.message,
        user_id: request.userId,
        preferred_agent: request.preferredAgent || 'auto',
        context: request.context,
    });
    return response.data;
}

/**
 * Check if multi-agent system is available
 */
export async function getAgentStatus(): Promise<{ status: string; agents: string[] }> {
    const response = await api.get('/agent/status');
    return response.data;
}

// ============= WebSocket 连接 (AI对话流式响应) =============

export function createChatWebSocket(
    userId: string,
    onMessage: (chunk: string) => void,
    onError: (error: Error) => void,
    onComplete: () => void
): WebSocket {
    const wsUrl = API_BASE_URL.replace('http', 'ws') + `/chat/stream?user_id=${userId}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.done) {
            onComplete();
        } else if (data.error) {
            onError(new Error(data.error));
        } else {
            onMessage(data.content || '');
        }
    };

    ws.onerror = () => {
        onError(new Error('WebSocket connection failed'));
    };

    return ws;
}

export default api;
