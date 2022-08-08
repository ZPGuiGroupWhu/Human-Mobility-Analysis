export function limitMemberCounts(input, value, max) {
  while (input.length >= max) {
    input.shift();
  }
  input.push(value);
  return input;
}
