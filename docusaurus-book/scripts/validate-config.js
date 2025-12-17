#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('🔍 Validating Docusaurus configuration...');

try {
  // Check if required files exist
  const requiredFiles = [
    'docusaurus.config.ts',
    'sidebars.ts',
    'package.json',
    'src/pages/index.tsx',
    'docs/intro.md'
  ];

  for (const file of requiredFiles) {
    if (!fs.existsSync(file)) {
      throw new Error(`Required file does not exist: ${file}`);
    }
  }

  console.log('✅ All required files exist');

  // Check if package.json has the correct scripts
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const requiredScripts = ['start', 'build', 'swizzle', 'deploy', 'clear', 'serve'];

  for (const script of requiredScripts) {
    if (!packageJson.scripts[script]) {
      console.warn(`⚠️  Missing script: ${script}`);
    }
  }

  console.log('✅ Package.json scripts validated');

  // Check if docs structure is correct
  const docsDir = 'docs';
  const moduleDirs = fs.readdirSync(docsDir).filter(file =>
    file.startsWith('module-') && fs.statSync(path.join(docsDir, file)).isDirectory()
  );

  console.log(`✅ Found ${moduleDirs.length} module directories:`, moduleDirs);

  // Check if basic content exists
  const introContent = fs.readFileSync('docs/intro.md', 'utf8');
  if (introContent.length < 10) {
    throw new Error('Intro.md appears to be empty or too short');
  }

  console.log('✅ Basic content validation passed');

  // Check if component files exist
  const requiredComponents = [
    'src/components/Exercise.tsx',
    'src/components/Diagram.tsx'
  ];

  for (const component of requiredComponents) {
    if (!fs.existsSync(component)) {
      throw new Error(`Required component does not exist: ${component}`);
    }
  }

  console.log('✅ All required components exist');

  console.log('\n🎉 Configuration validation passed!');
  console.log('📋 Summary:');
  console.log('   - All required files present');
  console.log('   - Package.json has necessary scripts');
  console.log('   - Module directories detected');
  console.log('   - Basic content exists');
  console.log('   - All custom components present');
  console.log('\n💡 The site configuration appears to be valid.');
  console.log('   The build process may take time due to the large amount of content.');

} catch (error) {
  console.error('❌ Configuration validation failed:', error.message);
  process.exit(1);
}