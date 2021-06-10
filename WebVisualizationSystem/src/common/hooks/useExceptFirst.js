import { useRef, useEffect } from 'react'

// 跳过首次渲染
export const useExceptFirst = (setFunc, ...data) => {
  const first = useRef(false);
  useEffect(() => {
    if (!first.current) {
      first.current = true;
    } else {
      setFunc(data)
    }
  }, [...data])
}
