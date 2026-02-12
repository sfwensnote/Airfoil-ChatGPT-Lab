/**
 * NACA 4位翼型几何生成库
 * 从Python移植到TypeScript，实现即时几何预览
 */

/**
 * NACA 4位翼型厚度分布公式
 */
export function naca4ThicknessDistribution(x: number[], t: number): number[] {
    return x.map(xi => {
        const xClip = Math.max(1e-12, Math.min(1.0, xi));
        return 5.0 * t * (
            0.2969 * Math.sqrt(xClip)
            - 0.1260 * xi
            - 0.3516 * xi ** 2
            + 0.2843 * xi ** 3
            - 0.1015 * xi ** 4
        );
    });
}

/**
 * 重映射x坐标以调整最大厚度位置
 */
export function remapXForTpos(
    x: number[],
    xDefaultPeak: number,
    xTargetPeak: number
): number[] {
    const x0 = Math.max(1e-6, Math.min(1 - 1e-6, xDefaultPeak));
    const xt = Math.max(1e-6, Math.min(1 - 1e-6, xTargetPeak));

    if (Math.abs(xt - x0) < 1e-8) return x;

    return x.map(xi => {
        let y: number;
        if (xi <= x0) {
            y = xi * (xt / x0);
        } else {
            y = xt + (xi - x0) * ((1.0 - xt) / (1.0 - x0));
        }
        return Math.max(0, Math.min(1, y));
    });
}

/**
 * 生成NACA 4位翼型坐标
 * @param m 弯度 (0-0.1)
 * @param p 最大弯度位置 (0-1)
 * @param t 厚度 (0.05-0.2)
 * @param tpos 最大厚度位置 (0-1)
 * @param nPts 点数
 */
export function genNaca4(
    m: number,
    p: number,
    t: number,
    tpos: number = 0.30,
    nPts: number = 200
): { x: number[]; y: number[] } {
    // 生成x坐标
    const x: number[] = [];
    for (let i = 0; i < nPts; i++) {
        x.push(i / (nPts - 1));
    }

    // 重映射x以调整厚度位置
    const xMapped = remapXForTpos(x, 0.30, tpos);

    // 计算厚度分布
    const yt = naca4ThicknessDistribution(xMapped, t);

    // 确保p在有效范围内
    const pSafe = Math.max(1e-6, Math.min(1 - 1e-6, p));

    // 计算中弧线
    const yc: number[] = [];
    const dycDx: number[] = [];

    for (let i = 0; i < nPts; i++) {
        const xi = x[i];
        if (xi < pSafe) {
            yc.push((m / (pSafe ** 2)) * (2 * pSafe * xi - xi ** 2));
            dycDx.push((2 * m / (pSafe ** 2)) * (pSafe - xi));
        } else {
            yc.push((m / ((1 - pSafe) ** 2)) * ((1 - 2 * pSafe) + 2 * pSafe * xi - xi ** 2));
            dycDx.push((2 * m / ((1 - pSafe) ** 2)) * (pSafe - xi));
        }
    }

    // 计算上下表面坐标
    const theta = dycDx.map(d => Math.atan(d));

    const xu: number[] = [];
    const yu: number[] = [];
    const xl: number[] = [];
    const yl: number[] = [];

    for (let i = 0; i < nPts; i++) {
        xu.push(x[i] - yt[i] * Math.sin(theta[i]));
        yu.push(yc[i] + yt[i] * Math.cos(theta[i]));
        xl.push(x[i] + yt[i] * Math.sin(theta[i]));
        yl.push(yc[i] - yt[i] * Math.cos(theta[i]));
    }

    // 合并为单一轮廓 (下表面反序 + 上表面)
    const xs = [...xl.reverse(), ...xu.slice(1)];
    const ys = [...yl.reverse(), ...yu.slice(1)];

    return { x: xs, y: ys };
}

/**
 * 根据参数生成NACA代码
 */
export function nacaCodeFromMpt(m: number, p: number, t: number): string {
    const mPct = Math.round(m * 100);
    const pTenths = Math.round(p * 10);
    const tPct = Math.round(t * 100);
    return `${mPct}${pTenths}${tPct.toString().padStart(2, '0')}`;
}

/**
 * 计算雷诺数
 */
export function estimateRe(rho: number, V: number, chord: number, mu: number): number {
    return (rho * V * chord) / Math.max(mu, 1e-12);
}

/**
 * 旋转坐标
 */
export function rotateXY(
    x: number[],
    y: number[],
    angleDeg: number,
    xc: number = 0.25,
    yc: number = 0
): { x: number[]; y: number[] } {
    const rad = (angleDeg * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    const xr: number[] = [];
    const yr: number[] = [];

    for (let i = 0; i < x.length; i++) {
        const dx = x[i] - xc;
        const dy = y[i] - yc;
        xr.push(xc + dx * cos - dy * sin);
        yr.push(yc + dx * sin + dy * cos);
    }

    return { x: xr, y: yr };
}
