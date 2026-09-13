//! P0630 — HTTP/1.1 server from `std::net` only: threads per connection,
//! keep-alive, Expect: 100-continue, body caps, read timeouts, panic isolation.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

const MAX_HEAD: usize = 16 * 1024;
const MAX_BODY: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
    pub extra_headers: Vec<(String, String)>,
}

impl Response {
    pub fn json(body: String) -> Response {
        Response {
            status: 200,
            content_type: "application/json".into(),
            body: body.into_bytes(),
            extra_headers: Vec::new(),
        }
    }
    pub fn binary(content_type: &str, body: Vec<u8>, filename: &str) -> Response {
        Response {
            status: 200,
            content_type: content_type.into(),
            body,
            extra_headers: vec![(
                "Content-Disposition".into(),
                format!("attachment; filename=\"{}\"", filename),
            )],
        }
    }
    pub fn text(status: u16, body: &str) -> Response {
        Response {
            status,
            content_type: "text/plain; charset=utf-8".into(),
            body: body.as_bytes().to_vec(),
            extra_headers: Vec::new(),
        }
    }
    pub fn not_found() -> Response {
        Response::text(404, "not found")
    }
}

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        100 => "Continue",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

/// Read one request from the stream. Returns None on EOF/timeout.
fn read_request(stream: &mut TcpStream) -> Option<Result<Request, String>> {
    // read until \r\n\r\n
    let mut head: Vec<u8> = Vec::with_capacity(1024);
    let mut buf = [0u8; 4096];
    let head_end;
    loop {
        if let Some(pos) = find_head_end(&head) {
            head_end = pos;
            break;
        }
        if head.len() > MAX_HEAD {
            return Some(Err("headers too large".into()));
        }
        let n = match stream.read(&mut buf) {
            Ok(0) => return None, // EOF
            Ok(n) => n,
            Err(_) => return None, // timeout or reset
        };
        head.extend_from_slice(&buf[..n]);
    }
    let head_str = String::from_utf8_lossy(&head[..head_end]).to_string();
    let mut lines = head_str.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    if method.is_empty() || path.is_empty() {
        return Some(Err("malformed request line".into()));
    }
    let mut headers: Vec<(String, String)> = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some(colon) = line.find(':') {
            let k = line[..colon].trim().to_string();
            let v = line[colon + 1..].trim().to_string();
            headers.push((k, v));
        }
    }
    let req_head = Request { method, path, headers, body: Vec::new() };
    // 100-continue
    if let Some(exp) = req_head.header("expect") {
        if exp.eq_ignore_ascii_case("100-continue") {
            let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
            let _ = stream.flush();
        }
    }
    // body
    let mut req = req_head;
    let len: usize = req
        .header("content-length")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    if len > MAX_BODY {
        return Some(Err("body too large (max 4 MB)".into()));
    }
    let mut body: Vec<u8> = head[head_end + 4..].to_vec();
    while body.len() < len {
        let n = match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        body.extend_from_slice(&buf[..n]);
    }
    body.truncate(len);
    req.body = body;
    Some(Ok(req))
}

fn find_head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn write_response(stream: &mut TcpStream, resp: &Response, keep_alive: bool) {
    let mut out = Vec::with_capacity(resp.body.len() + 256);
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\n",
        resp.status,
        status_text(resp.status),
        resp.content_type,
        resp.body.len(),
        if keep_alive { "keep-alive" } else { "close" },
    );
    out.extend_from_slice(head.as_bytes());
    for (k, v) in &resp.extra_headers {
        out.extend_from_slice(format!("{}: {}\r\n", k, v).as_bytes());
    }
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(&resp.body);
    let _ = stream.write_all(&out);
    let _ = stream.flush();
}

/// Serve forever. `handler` receives (method, path, body) and returns a Response.
pub fn serve(addr: &str, handler: Arc<dyn Fn(&Request) -> Response + Send + Sync>) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let handler = handler.clone();
        std::thread::spawn(move || {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
            let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
            loop {
                match read_request(&mut stream) {
                    None => break,
                    Some(Err(_)) => {
                        let resp = Response::text(400, "bad request");
                        write_response(&mut stream, &resp, false);
                        break;
                    }
                    Some(Ok(req)) => {
                        let keep = match req.header("connection") {
                            Some(c) => !c.eq_ignore_ascii_case("close"),
                            None => true,
                        };
                        // panic isolation: one bad request never kills the server
                        let resp = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            handler(&req)
                        }))
                        .unwrap_or_else(|_| Response::text(500, "internal error"));
                        write_response(&mut stream, &resp, keep);
                        if !keep {
                            break;
                        }
                    }
                }
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_logic() {
        // head-end finder
        assert_eq!(find_head_end(b"GET / HTTP/1.1\r\nHost: x\r\n\r\nBODY"), Some(23));
        assert_eq!(find_head_end(b"no end"), None);
    }
}
