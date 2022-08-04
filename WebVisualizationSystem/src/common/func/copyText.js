// 拷贝文本至剪贴板
// clipboardData.setData() 仅适用于 IE
// let res = window.clipboardData.setData("Text", message);

export function copyText(target, afterCopyCallback = null) {
  target.select();
  let res = document.execCommand('Copy', false, null);
  afterCopyCallback?.(res);
  return res;
}