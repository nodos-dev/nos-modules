#version 450

layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

layout(binding = 1) uniform UBO
{
    vec4 Color;
    uint Number;
    bool RenderFrameNo;
    bool SampleInput;
}
ubo;

layout(binding = 2) uniform sampler2D Input;

layout(push_constant) uniform constants
{
    uvec2 RTSize;
    uint frameNumber;
    float deltaSeconds;
} PC;

vec4 manhattan_distance(float i, float j, int b, vec2 R)
{
    // Computes a smooth-edged diamond pixel value (Manhattan distance)
	return vec4(b * smoothstep(0., 9. / R.y, .1 - abs(i) - abs(j)));
}

vec4 segment_value(float i, float j, int b, vec2 R)
{
    // Computes a segment value (length = 0.5)
	return manhattan_distance(i - clamp(i, 0., .5), j, b & 1, R);
}

void colon_render(float y, ivec4 i, vec2 R, inout float x, inout vec4 O, inout int t)
{
// Colon render

    x += .5;
    O += 
    manhattan_distance(
        x, 
        y + .3, 
        i.w / 50, 
        R
   ) 
   + manhattan_distance(
       x, 
       y - .3, 
       i.w / 50, 
       R
	);
    t /= 60;
}


vec4 s7_horrizontal(float i, float j, int b, float x, float y, vec2 R)
{
	// Computes the horizontal and vertical segments based on a denary digit
	return segment_value(x - i, y - j, b, R);
}

vec4 s7_vertical(float i, float j, int b, float x, float y, vec2 R)
{
	return segment_value(y - j, x - i, b, R);
}

void s7_segment(uint n, inout vec4 O, inout float x, float y, vec2 R)
{   
    // investigated with python:
	// {i: [bool((j>>i) & 1) for j in [892, 1005, 877, 881, 927, 325, 1019]] for i in range(10)}
    ++x; 
    O += segment_value(x, y, 892>>n, R)        // (1<<2 | 1<<3 | 1<<5 | 1 << 6 | 1<<8 | 1<<9)
                                               // 0b1101111100: true for 2,3,4,5,6,8,9
    + s7_horrizontal(0., .7, 1005>>n, x, y, R) // 0b1111101101: true for 0,2,3,5,6,7,8,9
    + s7_horrizontal(0., -.7, 877>>n, x, y, R) // 0b1101101101: true for 0,2,3,5,6,8,9
    + s7_vertical(-.1, .1, 881>>n, x, y, R)    // 0b1101110001: true for 0,4,5,6,8,9
    + s7_vertical(.6, .1, 927>>n, x, y, R)     // 0b1110011111: true for 0,1,2,3,4,7,8,9
    + s7_vertical(-.1, -.6, 325>>n, x, y, R)   // 0b0101000101: truw for 0,2,6,8
    + s7_vertical(.6, -.6, 1019>>n, x, y, R);  // 0b1111111011: truw for 0,1,3,4,5,6,7,8,9
                                               //    |   |   |
    										   //    |   |   \ zero
                                               //    |   |  \ one
                                               //    |   | \ two
                                               //    |   |\ three
                                               //    |   \ four
                                               //    |  \ five
                                               //    | \ six
                                               //    |\ seven
                                               //    \ eight
                                               //   \ nine
}


void main()
{
    vec2 R = vec2(PC.RTSize);
    uint N = mix(ubo.Number, uint(PC.frameNumber), ubo.RenderFrameNo);
    uint D = uint(log(N) / log(10)) + 1;

    float x = (uv.x - 1)  * 32 / 4 - .25 - D/2. + 4;
    float y = (.5 - uv.y) * 24 / 4;

    vec4 O = vec4(0);
    
    s7_segment(N % 10, O, x, y, R);
    N /= 10;
    
    while(N != 0)
    {
        s7_segment(N % 10, O, x, y, R);
        N /= 10;
    }

    rt = O * ubo.Color;
    if(ubo.SampleInput)
    {
        rt = mix(texture(Input, uv), ubo.Color, O);
    }

    if (uv.x < ((ubo.Number % 100) * 0.01))
        rt.x = 1;
}

