# Airfoil Lab - AI增强翼型设计实验室

一个现代化的翼型设计与学习平台，集成XFOIL仿真和AI辅导功能。

## 🚀 快速开始

### 本地运行

```bash
# 1. 进入项目目录
cd airfoil-lab-react

# 2. 安装依赖
npm install

# 3. 启动开发服务器
npm run dev

# 4. 访问 http://localhost:3000
```

### 同时运行后端（XFOIL仿真）

```bash
# 在另一个终端，进入父目录
cd bot-remote-windows

# 安装Python依赖
pip install -r requirements.txt

# 启动后端服务
uvicorn backend:app --host 0.0.0.0 --port 8000

# 确保 xfoil.exe 在同一目录下
```

## 📦 生产部署

### 方式1: 构建静态文件

```bash
# 构建生产版本
npm run build

# 启动生产服务器
npm start

# 或使用PM2进程管理
pm2 start npm --name "airfoil-lab" -- start
```

### 方式2: Docker部署

```dockerfile
# Dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

```bash
docker build -t airfoil-lab .
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://your-backend:8000 airfoil-lab
```

### 方式3: 部署到Vercel

```bash
npm i -g vercel
vercel
```

## 🔧 环境变量

创建 `.env.local` 文件：

```env
# 后端API地址（必须）
NEXT_PUBLIC_API_URL=http://localhost:8000

# 可选配置
NEXT_PUBLIC_ADMIN_PASS=your_admin_password
```

## 📁 项目结构

```
airfoil-lab-react/
├── src/
│   ├── app/                # Next.js 页面
│   │   ├── page.tsx        # 主页（几何+仿真）
│   │   ├── history/        # 历史记录页
│   │   ├── help/           # 帮助页
│   │   └── admin/          # 管理员页
│   ├── components/
│   │   ├── airfoil/        # 翼型相关组件
│   │   ├── chat/           # AI对话组件
│   │   ├── controls/       # 控制组件
│   │   ├── layout/         # 布局组件
│   │   └── ui/             # shadcn基础组件
│   ├── lib/
│   │   ├── geometry.ts     # 几何计算
│   │   ├── api.ts          # API客户端
│   │   └── utils.ts        # 工具函数
│   ├── stores/             # Zustand状态管理
│   └── types/              # TypeScript类型
├── public/                 # 静态资源
└── package.json
```

## ⚙️ 后端API端点

| 端点 | 方法 | 描述 |
|-----|------|------|
| `/simulate` | POST | 运行XFOIL仿真 |
| `/save_airfoil/` | POST | 保存翼型数据 |
| `/export_airfoils/{user_id}` | GET | 获取用户历史 |
| `/save_conversation/` | POST | 保存对话 |
| `/export_conversations/{user_id}` | GET | 获取对话历史 |
| `/admin/export_all_*` | GET | 管理员导出 |

## 🛠️ 开发命令

```bash
npm run dev      # 开发模式
npm run build    # 构建
npm run start    # 生产模式
npm run lint     # 代码检查
```

## 📄 License

MIT
