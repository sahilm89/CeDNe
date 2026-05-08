const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('http://localhost:5173');
  await page.waitForTimeout(2000); // Wait for React to render
  const header = await page.$('.header-panel');
  if (header) {
    const box = await header.boundingBox();
    console.log('Header bounding box:', box);
  } else {
    console.log('Header not found');
  }
  const title = await page.$('.graph-title');
  if (title) {
    const box = await title.boundingBox();
    console.log('Title bounding box:', box);
  }
  const logo = await page.$('img[alt="CeDNe"]');
  if (logo) {
    const box = await logo.boundingBox();
    console.log('Logo bounding box:', box);
  }
  await browser.close();
})();
