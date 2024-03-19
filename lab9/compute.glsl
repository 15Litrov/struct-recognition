#version 430 core

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;


layout(rgba32f, binding = 0) uniform image2D imgIn;
layout(rgba32f, binding = 1) uniform image2D imgMask;
layout(rgba32f, binding = 2) uniform image2D imgOutput;

uniform ivec2 offset;
uniform ivec2 size;

void main() {
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);

    if (texelCoord.x == 0 || texelCoord.x >= size.x - 1 || texelCoord.y == 0 || texelCoord.y >= size.y - 1)
        return;

    if (imageLoad(imgMask, texelCoord).r < 0.5)
        return;

    vec4 B = imageLoad(imgIn, texelCoord);

    ivec2 c = texelCoord + offset;
    ivec2 l = c + ivec2(-1, -1);
    ivec2 r = c + ivec2(-1, 1);
    ivec2 t = c + ivec2(1, 1);
    ivec2 b = c + ivec2(1, -1);

    vec4 U = -imageLoad(imgOutput, l) - imageLoad(imgOutput, r) - imageLoad(imgOutput, b) - imageLoad(imgOutput, t);
	
    barrier();

    imageStore(imgOutput, c, B - U * 0.25);

    memoryBarrierShared();
    barrier();

    l = c + ivec2(-1, 0);
    r = c + ivec2(1, 0);
    t = c + ivec2(0, 1);
    b = c + ivec2(0, -1);

    U = -imageLoad(imgOutput, l) - imageLoad(imgOutput, r) - imageLoad(imgOutput, b) - imageLoad(imgOutput, t);

    barrier();

    imageStore(imgOutput, c, B - U * 0.25);
}