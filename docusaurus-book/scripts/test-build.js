#!/usr/bin/env node

const { execSync } = require('child_process');
const path = require('path');

console.log('🧪 Testing Docusaurus build for Physical AI & Humanoid Robotics Textbook...');

try {
  // Change to the docusaurus directory
  const docusaurusDir = path.resolve(__dirname, '..');
  process.chdir(docusaurusDir);

  console.log('📦 Installing dependencies...');
  execSync('npm ci', { stdio: 'inherit' });

  console.log('🔨 Building the site...');
  execSync('npm run build', { stdio: 'inherit' });

  console.log('✅ Build completed successfully!');
  console.log('✅ All modules and content built without errors');
  console.log('✅ Site ready for deployment');

  // Test that the build directory exists and has content
  const fs = require('fs');
  const buildDir = path.join(docusaurusDir, 'build');
  if (fs.existsSync(buildDir) && fs.readdirSync(buildDir).length > 0) {
    console.log('✅ Build directory contains generated content');
  } else {
    throw new Error('Build directory is empty or does not exist');
  }

  console.log('\n🎉 Build validation passed!');
  console.log('📋 Summary:');
  console.log('   - Dependencies installed successfully');
  console.log('   - Site built without errors');
  console.log('   - Generated content verified');
  console.log('   - Ready for deployment');

} catch (error) {
  console.error('❌ Build test failed:', error.message);
  process.exit(1);
}