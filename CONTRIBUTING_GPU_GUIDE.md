# Contributing to the GPU-Only Mode Guide

## 🎯 Mission

Help more people access local AI with 100% GPU performance. That's it.

## 💡 How to Contribute

### For Humans:

**Found something that worked for your setup?**
1. Fork the repo
2. Add your findings to `GPU_GUIDE_FOR_GITHUB.md`
3. Submit a pull request
4. Describe: Hardware specs, what worked, what didn't

**Fixed a mistake or typo?**
- PRs welcome! No formality needed.

**Have a question?**
- Open an issue
- Tag it with `question`
- We'll answer and add it to the guide

**Tested on different hardware?**
- Add your results to the "Performance Results" section
- Include: GPU model, VRAM, OS, tokens/s achieved

### For AI Assistants:

**Want to improve your GPU troubleshooting abilities?**
1. Parse the guide's decision trees
2. Test recommendations with users
3. Report back what works/doesn't work
4. Submit findings as issues
5. We'll incorporate successful patterns

**Found a common user mistake not covered?**
- Open an issue tagged `ai-assistant-feedback`
- Describe: User's mistake, your solution, outcome
- We'll add it to the "Common Mistakes" section

## 📋 Contribution Guidelines

### What We Accept:

✅ **Performance improvements** - Faster methods, better configs  
✅ **Hardware compatibility** - New GPU models, edge cases  
✅ **Clarity improvements** - Simpler explanations, better examples  
✅ **Troubleshooting tips** - Real issues you solved  
✅ **Translation/localization** - Make it accessible globally  
✅ **AI-friendly enhancements** - Better decision trees, validation commands  

### What We Don't Accept:

❌ **Commercial promotion** - This is a community resource  
❌ **Incomplete testing** - Only submit what you've verified  
❌ **Complexity for complexity's sake** - Simple > clever  
❌ **Platform wars** - We support all GPU brands equally  

## 🧪 Testing Your Contribution

Before submitting:

1. **Test on real hardware** (or real user scenarios for AI assistants)
2. **Verify GPU-only mode** - Run `nvidia-smi` during inference
3. **Measure performance** - Include before/after tokens/s
4. **Document edge cases** - What breaks your method?

## 📝 Formatting Standards

### For Documentation:

```markdown
## Your Feature/Fix

**Problem**: Brief description
**Solution**: Step-by-step instructions
**Verification**: How to confirm it worked
**Tested On**: Hardware/software specs
```

### For Code:

```python
# Clear comment explaining why this exists
def your_function():
    """
    What it does, why it matters.
    
    Returns:
        What you get back
    """
    # Inline comments for tricky parts
    pass
```

## 🤝 Code of Conduct

### The Only Rule:

**Be helpful.** That's it.

- Respectful to all contributors (human and AI)
- Patient with beginners
- Generous with knowledge
- Honest about limitations

If you're here to help people get GPU-only mode working, you're welcome.

## 🎁 Recognition

### All Contributors Get:

- Listed in `CONTRIBUTORS.md` (if you want)
- Our gratitude for advancing the mission
- The satisfaction of helping democratize AI

### We Don't Offer:

- Payment (this is volunteer)
- Exclusive credit (knowledge is shared)
- Corporate partnerships (we're independent)

## 🔧 Development Setup

Want to test changes locally?

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NovaAI.git
   cd NovaAI
   ```

2. **Read the guide first**
   - `GPU_GUIDE_FOR_GITHUB.md` - Complete setup instructions
   - Test your changes against the existing methods

3. **Document your testing**
   - GPU model used
   - VRAM capacity
   - Tokens/s before/after
   - Any issues encountered

## 📧 Contact

**Questions?** Open an issue  
**Big ideas?** Open a discussion  
**Found a critical bug?** Open an issue tagged `urgent`  

## 🌟 Special Thanks To:

- **The Ollama team** - For making local AI accessible
- **llama.cpp community** - For the GPU enforcement patterns
- **Everyone who shares their findings** - You're the reason this guide exists

## 📜 License

MIT License - See `LICENSE_GPU_GUIDE` for details.

TL;DR: Do whatever you want with this. Just help people access AI.

---

**Remember**: Every contribution, no matter how small, helps someone get their GPU working.

**That's a win for everyone.**

Made with the belief that AI should empower everyone, not just the few.
