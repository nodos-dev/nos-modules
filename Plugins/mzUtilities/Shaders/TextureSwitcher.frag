#version 450

layout(binding = 0) uniform sampler2D Channel_1;
layout(binding = 1) uniform sampler2D Channel_2;
layout(binding = 2) uniform sampler2D Channel_3;
layout(binding = 3) uniform sampler2D Channel_4;
layout(binding = 4) uniform sampler2D Channel_5;
layout(binding = 5) uniform sampler2D Channel_6;
layout(binding = 6) uniform sampler2D Channel_7;
layout(binding = 7) uniform sampler2D Channel_8;
layout(binding = 8) uniform sampler2D Channel_9;
layout(binding = 9) uniform sampler2D Channel_10;

layout(binding = 11) uniform TextureSwitcherParams
{
    uint Output_Channel;
} Params;


layout(location = 0) out vec4 rt;
layout(location = 0) in vec2 uv;

void main()
{
    switch(Params.Output_Channel)
    {
        case 0: rt = texture(Channel_1, uv); break;
        case 1: rt = texture(Channel_2, uv); break;
        case 2: rt = texture(Channel_3, uv); break;
        case 3: rt = texture(Channel_4, uv); break;
        case 4: rt = texture(Channel_5, uv); break;
        case 5: rt = texture(Channel_6, uv); break;
        case 6: rt = texture(Channel_7, uv); break;
        case 7: rt = texture(Channel_8, uv); break;
        case 8: rt = texture(Channel_9, uv); break;
        case 9: rt = texture(Channel_10, uv); break;
        default: rt = vec4(1.0, 0.0, 0.0, 1.0); break;
    }
}
